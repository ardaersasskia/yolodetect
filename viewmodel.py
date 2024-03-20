import netron
import torchsummary as summary
import torch
# 加载模型
model = torch.hub.load('./', 'custom', './pretrained/balanced200.pt',source='local', force_reload=False)
#print(model)
total_params = sum(p.numel() for p in model.parameters())
total_params += sum(p.numel() for p in model.buffers())
print(f'{total_params:,} total parameters.')
print(f'{total_params/(1024*1024):.2f}M total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
#使用Netron可视化模型
#netron.start()