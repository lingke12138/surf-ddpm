import torch
from torch import nn, optim
import numpy as np
from tqdm.auto import tqdm
import os
from Trainer.Trainer_utils import EarlyStopping, TrainerBase
from Dataset.surf_data_loader import surf_data_loader

from Model.GaussianDiffusion import get_named_beta_schedule
from Model.Unet_cat_label import Text2ImUNet
from Model.GaussianDiffusion import GaussianDiffusion
from torch.optim import Adam, RAdam, Adadelta, NAdam, RMSprop, SGD
import matplotlib.pyplot as plt
import matplotlib
import csv
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")

# 调用数据
path = r'F:\表压'
train_pressure = surf_data_loader(path, 'train')
train_loader = torch.utils.data.DataLoader(train_pressure, batch_size=64, shuffle=True, pin_memory=True, num_workers=0,drop_last=True)
val_pressure = surf_data_loader(path, 'val')
val_loader = torch.utils.data.DataLoader(val_pressure, batch_size=64, shuffle=True, pin_memory=True,num_workers=0, drop_last=True)
test_pressure = surf_data_loader(path, 'val')
test_loader = torch.utils.data.DataLoader(test_pressure, batch_size=64, shuffle=True, pin_memory=True,num_workers=0, drop_last=True)

# 导入训练好的模型
# 设置参数
image_size = 32  # 我的数据转成图片样式之后reshape后的图片尺寸
in_channels = 5     # 我的数据转成图片样式之后reshape后的channel数
model_channels = 128        # 神经网络模型中的通道数
out_channels = 5
num_res_blocks = 3     # 残差块的数量，指示模型中包含的残差块的数量
attention_resolutions = "16,8,4"     # 注意力分辨率，表示模型在注意力操作中使用的分辨率
attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(image_size // int(res))
channel_mult = "1,2,4"
channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
assert 2 ** (len(channel_mult) + 2) == image_size
use_fp16 = False
num_heads = 1
num_head_channels = 64
num_heads_upsample = -1
use_scale_shift_norm = True
resblock_updown = True
betas = get_named_beta_schedule(schedule_name="linear", num_diffusion_timesteps=1000)
schedule_name="linear"
# 模型搭建
Unet = Text2ImUNet(text_ctx=64, xf_width=128, xf_layers=8, xf_heads=2, xf_final_ln=True,
                   input_embedding_seqnum=30, cache_text_emb=True, share_unemb=True,
                   in_channels=in_channels, model_channels=model_channels, out_channels=out_channels,
                   num_res_blocks=num_res_blocks, attention_resolutions=tuple(attention_ds),
                   dropout=0.1, channel_mult=channel_mult, use_fp16=use_fp16, num_heads=num_heads,
                   num_head_channels=num_head_channels, num_heads_upsample=num_heads_upsample,
                   use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown,
                   num_classes=None, dims=2, conv_resample=True, use_checkpoint=False)
device = torch.device('cuda:0')
DDPM = GaussianDiffusion(betas=betas, model=Unet).to(device)
# DDPM_1 = DDPM(mode='train')
epoches = 10000
# 模型存储路径
root_path = r"F:\PycharmProjects\PycharmProjects\DDPM_label2img\Trainer\Model_save_package"
setting = "Adadelta_imageSize{}_channels{}_timeSteps{}_epoch{}_scheduleName{}".format(image_size, in_channels, len(betas.shape), epoches, schedule_name)
saved_path = os.path.join(root_path, setting)
checkpoint_path = saved_path +'/'+'BestModel.pth'
# 导入模型
checkpoint = torch.load(checkpoint_path)
DDPM.load_state_dict(checkpoint['model_state_dict'])


# 模型计算结果预处理（以用于p_sample）
guidance_scale = 1
def model_fn(x_t, ts, labels):
    half = x_t[: len(x_t) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = DDPM.model(combined, ts, labels)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    # print(eps.shape)
    return torch.cat([eps, rest], dim=1)

# 生成用于sample的数据集
val_loop = tqdm(enumerate(val_loader), total=len(val_loader))
for val_steps, (val_features, val_labels) in val_loop:
    val_features = val_features.view(len(val_features), 5, 32, 32).to(device)
    val_labels = val_labels.view(len(val_labels), 30).to(device)
    val_batch_size = val_features.shape[0]
    t = torch.randint(0, 1000, (val_batch_size,), device=device).long()
    # 开始sample
    noise = torch.randn_like(val_features)
    pred_noise = DDPM.model(noise, t, val_labels)
    samples, shape = DDPM.p_sample_loop((64, 5, 32, 32), noise=noise, device=device,
                                 clip_denoised=True, progress=True, label=val_labels)
    for i in range(64):
        print(val_features[i].shape)
        print(samples[i].shape)
        plt.hist(val_features[i].cpu().numpy().flatten(), color='red', edgecolor='black', label='noise', bins=10)
        plt.hist(samples[i].cpu().numpy().flatten(), color='blue', edgecolor='black', label='pred', bins=10)
        plt.legend()
        plt.xlabel('pressure')
        plt.ylabel('Frequency')
        # plt.title('Histogram with Legend')
        plt.savefig(f'sample\\group{val_steps}pic{i}.jpg', format='jpg', dpi=200)
        plt.close('all')
        with open('sample\\val_label.csv', mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            # writer.writerow((['epoch', 'step', 'predicted_noise_mean', 'predicted_noise_std', 'noise_mean', 'noise_std']))
            text = [val_steps, val_labels[i].tolist()]
            print(text)
            writer.writerow(text)