import torch

import os
import sys
import json
import math
import copy
import random
import traceback
import transformers
import numpy as np
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_submodules
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
except ImportError:
    print("Failed to import qwen2_vl; Please update it transformers to 4.45.0`")

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError

from qmllm.models.base import BaseModel
from qmllm.utils.registry import MODEL_REGISTRY

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

@MODEL_REGISTRY.register("qwen2_vl")
class Qwen2_VL(BaseModel):
    def __init__(self, model, tokenizer, processor):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.device_map = getattr(model, 'hf_device_map', {})

    def fetch_vit(self):
        return self.model.vision_model

    def fetch_llm(self):
        return self.model.language_model

    def fetch_proj(self):
        return self.model.mlp1

    def vision_preprocess(self, image):
        pass
      
    def language_preprocess(self, text):
        return self.tokenizer(text)

    
    def forward(
            self, 
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')

        outputs = self.model.model(
            input_ids=input_ids.to(next(self.model.parameters()).device) if input_ids is not None else None,
            attention_mask=attention_mask.to(next(self.model.parameters()).device) if attention_mask is not None else None,
            inputs_embeds=inputs_embeds.to(next(self.model.parameters()).device) if inputs_embeds is not None else None,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=None,
        )


    def to_cuda(self):
        if self.num_params > 20 * 10 ** 9: # 20B model
            self.model = dispatch_model(self.model, device_map=self.device_map)
        else:
            self.model = self.model.cuda()

    def to_cpu(self):
        if self.num_params > 20 * 10 ** 9: # 20B model
            remove_hook_from_submodules(self.model)
        self.model = self.model.cpu()


    def convert_data_item(self, data_item):
        conversations = data_item["conversations"]
        for conv in conversations:
            if conv["from"] == "human":
                user_text = conv["value"]
                if "<image>" in user_text:
                    user_text = user_text.replace("<image>", "")
                if "\n" in user_text:
                    user_text = user_text.replace("\n", "")
            if conv["from"] == "gpt":
                asst_text = conv["value"]
        # Actually we don't use the image path here
        image_path = data_item["image"]
        item = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": asst_text,
            },
        ]

        return item
        

    def preprocess_data(self, images, data_item):
        new_data_item = self.convert_data_item(data_item)
        data_item_text = self.processor.apply_chat_template(
            new_data_item, tokenize=False, add_generation_prompt=False,
        )

        data_dict = self.processor(
            text=data_item_text,
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = torch.squeeze(v)

        # create labels
        labels = copy.deepcopy(data_dict["input_ids"])
        start_token_id = self.tokenizer.encode("<|im_start|>")[0]
        ans_start_indice = torch.where(labels == start_token_id)[0][-1].item()

        labels[:ans_start_indice] = -100

        data_dict["labels"] = labels

        return data_dict


    @torch.no_grad() 
    def few_shot_data_samples(self, data_samples, pad_side="left", interleave_freq=2):
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        pixel_values = data_samples["pixel_values"]
        image_grid_thw = data_samples["image_grid_thw"]

        # process input_ids, labels, attention_mask
        new_input_ids = []
        new_labels = []

        input_ids = [cur_input_ids[cur_attention_mask.bool()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask.bool()] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        for i in range(0, len(input_ids) - interleave_freq + 1, interleave_freq):
            cur_input_ids = input_ids[i: i+interleave_freq]
            concat_input_ids = torch.cat(cur_input_ids, dim=0)
            new_input_ids.append(concat_input_ids)

        for i in range(0, len(labels) - interleave_freq + 1, interleave_freq):
            cur_labels = labels[i: i+interleave_freq]
            concat_labels = torch.cat(cur_labels, dim=0)
            new_labels.append(concat_labels)

        # maybe we should add some codes to ensure the sequences we make are not longer than tokenizer max length
        # but now don't care it
         
        max_len = max(x.shape[0] for x in new_input_ids)
        batch_size = len(new_input_ids)
        new_input_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

        for i, (cur_new_input_ids, cur_new_labels) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = cur_new_input_ids.shape[0]
            if pad_side == "left":
                if cur_len > 0:
                    new_input_ids_padded[i, -cur_len:] = cur_new_input_ids
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_attention_mask[i, -cur_len:] = True
            else:
                if cur_len > 0:
                    new_input_ids_padded[i, :cur_len] = cur_new_input_ids
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_attention_mask[i, :cur_len] = True
                    
        new_data_samples = {}
        new_data_samples["input_ids"] = new_input_ids_padded
        new_data_samples["labels"] = new_labels_padded
        new_data_samples["attention_mask"] = new_attention_mask
        new_data_samples["pixel_values"] = pixel_values
        new_data_samples["image_grid_thw"] = image_grid_thw

        return new_data_samples


    @torch.no_grad() 
    def interleave_data_samples(self, data_samples, pure_text=None, pad_side="left", interleave_freq=2):
        
        input_ids = data_samples["input_ids"]
        labels = data_samples["labels"]
        attention_mask = data_samples["attention_mask"]
        pixel_values = data_samples["pixel_values"]
        image_grid_thw = data_samples["image_grid_thw"]

        # process input_ids, labels, attention_mask
        new_input_ids = []
        new_labels = []

        input_ids = [cur_input_ids[cur_attention_mask.bool()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask.bool()] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        for i in range(0, len(input_ids) - interleave_freq + 1, interleave_freq):
            cur_input_ids = [input_ids[i]]

            for j in range(interleave_freq-1):
                cur_input_ids.append(pure_text[i+j])
                cur_input_ids.append(input_ids[i+1+j])

            concat_input_ids = torch.cat(cur_input_ids, dim=0)
            new_input_ids.append(concat_input_ids)

        for i in range(0, len(labels) - interleave_freq + 1, interleave_freq):
            cur_labels = [labels[i]]

            for j in range(interleave_freq-1):
                cur_labels.append(pure_text[i+j])
                cur_labels.append(labels[i+1+j])
                
            concat_labels = torch.cat(cur_labels, dim=0)
            new_labels.append(concat_labels)

        # maybe we should add some codes to ensure the sequences we make are not longer than tokenizer max length
        # but now don't care it
         
        max_len = max(x.shape[0] for x in new_input_ids)
        batch_size = len(new_input_ids)
        new_input_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), -100, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

        for i, (cur_new_input_ids, cur_new_labels) in enumerate(zip(new_input_ids, new_labels)):
            cur_len = cur_new_input_ids.shape[0]
            if pad_side == "left":
                if cur_len > 0:
                    new_input_ids_padded[i, -cur_len:] = cur_new_input_ids
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_attention_mask[i, -cur_len:] = True
            else:
                if cur_len > 0:
                    new_input_ids_padded[i, :cur_len] = cur_new_input_ids
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_attention_mask[i, :cur_len] = True
                    
        new_data_samples = {}
        new_data_samples["input_ids"] = new_input_ids_padded
        new_data_samples["labels"] = new_labels_padded
        new_data_samples["attention_mask"] = new_attention_mask
        new_data_samples["pixel_values"] = pixel_values
        new_data_samples["image_grid_thw"] = image_grid_thw

        return new_data_samples

    @torch.no_grad()   
    def generate_input(self, data_samples):
        input_ids = data_samples['input_ids'].cuda()
        attention_mask = data_samples['attention_mask'].cuda()
        labels = data_samples['labels'].cuda() 
        pixel_values = data_samples['pixel_values'].to(self.model.dtype).cuda()
        image_grid_thw = data_samples['image_grid_thw'].cuda()

        # generate input embeddings
        # copied from the Qwen2VLForConditionalGeneration.forward
        inputs_embeds = self.model.model.embed_tokens(input_ids) 
        pixel_values = pixel_values.type(self.model.visual.get_dtype())
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        vision_mask = (input_ids == self.model.config.image_token_id)
        answer_mask = (labels != -100) # ignore token id

        prompt_inputs = {
            "inputs_embeds": inputs_embeds
        }

        prompt_kwargs = {
            "labels": labels,
            "attention_mask": attention_mask,
            "vision_mask": vision_mask,
            "caption_mask": answer_mask
        }

        return prompt_inputs, prompt_kwargs
        

    def data_collator(self, instances):
        # qwen2_vl's padding side is left
        pad_id = self.tokenizer.pad_token_id
        IGNORE_INDEX = -100
        first = instances[0]
        batch = {}

        batch_lens = [feat['input_ids'].shape for feat in instances]
        max_item_length = max(batch_lens)[0]
        for idx in range(len(instances)):
            feat = instances[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            # temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            temp_input_ids[-feat['input_ids'].shape[0]:] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            # temp_labels[:feat['labels'].shape[0]] = feat['labels']
            temp_labels[-feat['labels'].shape[0]:] = feat['labels']
            feat['labels'] = temp_labels
            feat['attention_mask'] = feat['input_ids'].ne(pad_id).int()

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if 'label' in first and first['label'] is not None:
            label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
            dtype = torch.long if isinstance(label, int) else torch.float
            batch['labels'] = torch.tensor([f['label'] for f in instances], dtype=dtype)
        elif 'label_ids' in first and first['label_ids'] is not None:
            if isinstance(first['label_ids'], torch.Tensor):
                batch['labels'] = torch.stack([f['label_ids'] for f in instances])
            else:
                dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
                batch['labels'] = torch.tensor([f['label_ids'] for f in instances], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('label', 'pixel_values', 'image_grid_thw') and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.tensor([f[k] for f in instances])
            if k in ('pixel_values'):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.concat([f[k] for f in instances])
            if k in ('image_grid_thw'):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in instances])
            if k in ('sample_id'):
                batch[k] = [f[k] for f in instances]
        return batch
