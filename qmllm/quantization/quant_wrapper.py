import os

from qmllm.methods.awq.entry import awq_entry
from qmllm.methods.smoothquant.entry import smoothquant_entry
from qmllm.methods.mbq.entry import mbq_entry
from qmllm.methods.rtn.entry import rtn_entry

def qwrapper(model, prompt_inputs, prompt_kwargs, args):
    wa_quant = args.w_bit < 16 and args.a_bit < 16
    model = mbq_entry(model, prompt_inputs, prompt_kwargs, 
                            run_mbq_process=args.run_process, 
                            pseudo_quant=args.pseudo_quant, 
                            scale_path=args.scale_path, 
                            q_group_size=args.w_group, 
                            w_bit=args.w_bit, 
                            a_bit=args.a_bit, 
                            wa_quant=wa_quant, 
                            reweight=args.reweight,
                            distort=args.distort,
                            loss_mode=args.loss_mode)
    return model