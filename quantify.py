class VARGradHook:
    def __init__(self):
        self.hooks = []
        self.grad_stats = defaultdict(lambda: {
            'attn_mat_qkv': [],
            'attn_proj': [],
            'ffn_fc1': [],
            'ffn_fc2': []
        })

    def _grad_hook(self, module, grad_input, grad_output, layer_idx, module_type):
        # 获取梯度张量
        grad = grad_output[0]
        
        # 计算梯度统计量
        stats = {
            'max': grad.abs().max().item(),
            'mean': grad.abs().mean().item(),
            'std': grad.std().item()
        }
        
        # 根据模块类型存储
        key = f"layer{layer_idx}_{module_type}"
        if 'mat_qkv' in module._get_name():
            self.grad_stats[key]['attn_mat_qkv'].append(stats)
        elif 'proj' in module._get_name():
            self.grad_stats[key]['attn_proj'].append(stats)
        elif 'fc1' in module._get_name():
            self.grad_stats[key]['ffn_fc1'].append(stats)
        elif 'fc2' in module._get_name():
            self.grad_stats[key]['ffn_fc2'].append(stats)

    def register_hooks(self, var_model):
        # 遍历所有AdaLNSelfAttn模块
        for layer_idx, block in enumerate(var_model.blocks):
            # 注册attn模块的钩子
            attn = block.attn
            self.hooks.append(attn.mat_qkv.register_full_backward_hook(
                functools.partial(self._grad_hook, layer_idx=layer_idx, module_type='attn')
            ))
            self.hooks.append(attn.proj.register_full_backward_hook(
                functools.partial(self._grad_hook, layer_idx=layer_idx, module_type='attn')
            ))
            
            # 注册ffn模块的钩子
            ffn = block.ffn
            self.hooks.append(ffn.fc1.register_full_backward_hook(
                functools.partial(self._grad_hook, layer_idx=layer_idx, module_type='ffn')
            ))
            self.hooks.append(ffn.fc2.register_full_backward_hook(
                functools.partial(self._grad_hook, layer_idx=layer_idx, module_type='ffn')
            ))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def print_grad_stats(self):
        for layer_name, modules in self.grad_stats.items():
            print(f"===== {layer_name} =====")
            for module_type, stats_list in modules.items():
                if stats_list:
                    avg_max = np.mean([s['max'] for s in stats_list])
                    avg_mean = np.mean([s['mean'] for s in stats_list])
                    avg_std = np.mean([s['std'] for s in stats_list])
                    print(f"{module_type}:")
                    print(f"  Max Grad: {avg_max:.4e}")
                    print(f"  Mean Grad: {avg_mean:.4e}") 
                    print(f"  Std Grad: {avg_std:.4e}")