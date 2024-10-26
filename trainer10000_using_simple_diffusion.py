import torch
from torch import nn, optim
import numpy as np
from tqdm.auto import tqdm
import os
from Trainer.Trainer_utils import EarlyStopping, TrainerBase
from Dataset.surf_data_loader import surf_data_loader
from Model.simpleDiffusion import DiffusionModel
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



# 设置参数
betas = get_named_beta_schedule(schedule_name="linear", num_diffusion_timesteps=1000)
schedule_name="linear"
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
DDPM = DiffusionModel(schedule_name='linear_beta_schedule', timesteps=1000, beta_start=0.0001, beta_end=0.002, denoise_model=Unet).to(device)
# DDPM_1 = DDPM(mode='train')

# 设置trainer
class DDPM_Trainer(TrainerBase):
    def __init__(self,start=None,epochs=None,train_loader=None,val_loader=None,optimizer=None,device=None,IFEarlyStopping=False,
                 IFadjust_learning_rate=False,**kwargs):
        super(DDPM_Trainer, self).__init__(epochs, train_loader, val_loader, optimizer, device, IFEarlyStopping, IFadjust_learning_rate, **kwargs)
        '''
        if 'timesteps' in kwargs.keys():
            self.timesteps = kwargs['timesteps']
        else:
            raise ValueError('扩散模型训练必须提供扩散步数参数')
    '''
        self.start = start


    def forward(self, model, *args, **kwargs):
        tensorboard_writer = SummaryWriter('logs')
        indices = torch.linspace(self.start, self.epochs+self.start-1, self.epochs, dtype=int)
        for i in indices:
            model.train()
            losses = []
            train_loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

            for steps, (features, labels) in train_loop:
                features = features.view(len(features), 5, 32, 32).to(self.device)
                labels = labels.view(len(labels), 30).to(self.device)

                batch_size = features.shape[0]
                t = torch.randint(0, 1000, (batch_size,), device=self.device).long()

                loss, noise, pred_noise = model(mode='train', x_start=features, t=t, label=labels, loss_type='l2')

                losses.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                x_noise = model.q_sample(x_start=features, t=t, noise=torch.randn_like(features))
                pred = Unet(x_noise, t, labels)
                # print('loss', (x_noise - pred).max().item())
                # with open(f'parameter_gradients.txt', 'w') as f:
                    # 遍历模型中每个参数，并将参数名称和梯度写入文件
                    # for name, param in model.named_parameters():
                        # if param.grad is not None:
                        # 将参数名称和梯度写入文件
                        #f.write(f'epoch:{i}, steps:{steps}, arameter: {name}, Gradient: {param.grad}\n')
                # for name, param in model.named_parameters():
                    # print(f'Parameter: {name}, Gradient: {param.grad}')


                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()

                        if grad_norm > 0.1:  # 设定梯度最大阈值
                            # 进行相应处理，例如打印警告信息或者梯度裁剪
                            print("梯度爆炸，需要进行处理")
                            print(grad_norm)

                # 更新信息
                train_loop.set_description(f'Epoch[{i}/{self.epochs+self.start-1}]')
                train_loop.set_postfix(loss=loss.item())
            tensorboard_writer.add_scalar('Loss/train', loss, i)
            ckpt_path = os.path.join(r'F:\PycharmProjects\PycharmProjects\DDPM_label2img\Trainer\Model_save_package\Adadelta_imageSize32_channels5_timeSteps1_epoch10000_scheduleNamelinear', f'checkpoint_{i}.pth')
            if i % 200 == 0:
                print(i)
                torch.save({'epoch': i, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
                print(ckpt_path)
            # 验证阶段
            model.eval()
            val_losses = []
            best_val_loss = 2.1
            with torch.no_grad():
                val_loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
                for val_steps, (val_features, val_labels) in val_loop:
                    val_features = val_features.view(len(val_features), 5, 32, 32).to(self.device)
                    val_labels = val_labels.view(len(val_labels), 30).to(self.device)

                    val_batch_size = val_features.shape[0]
                    t = torch.randint(0, 1000, (val_batch_size,), device=self.device).long()

                    val_loss, val_noise, val_pred_noise = model(mode='eval', x_start=val_features, t=t,
                                                                label=val_labels,
                                                                loss_type='l2')

                    val_pred = val_pred_noise.detach()
                    val_real = val_noise.detach()
                    if val_steps % 25 == 0:
                        plt.hist(val_real.cpu().numpy().flatten(), color='red', label='noise', bins=5)
                        plt.hist(val_pred.cpu().numpy().flatten(), color='blue', label='pred', bins=5)
                        plt.savefig(f'adadelta-2000\\epoch{i}step{steps}.jpg', format='jpg', dpi=200)
                        plt.close('all')
                    pred_mean_std = [i, steps, val_pred.mean().item(), val_pred.std().item(), val_noise.mean().item(),
                                     val_noise.std().item()]
                    with open('adadelta_val_pred_noise.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        # writer.writerow((['epoch', 'step', 'predicted_noise_mean', 'predicted_noise_std', 'noise_mean', 'noise_std']))
                        writer.writerow(pred_mean_std)

                    val_losses.append(val_loss.item())

                    val_loop.set_description(f'Validation')
                    val_loop.set_postfix(val_loss=val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            tensorboard_writer.add_scalar('Loss/val', avg_val_loss, i)
            self.early_stopping(avg_val_loss, epoch=i, optimizer=optimizer, model=model, path=kwargs['model_save_path'])
            if i % 200 == 0:
                print(i)
                val_loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
                for val_steps, (val_features, val_labels) in val_loop:
                    val_features = val_features.view(len(val_features), 5, 32, 32).to(self.device)
                    val_labels = val_labels.view(len(val_labels), 30).to(self.device)

                    val_batch_size = val_features.shape[0]
                    t = torch.randint(0, 1000, (val_batch_size,), device=self.device).long()

                    samples = model(mode="generate", labels=val_labels[1], image_size=image_size, batch_size=16,
                                    channels=in_channels)
                    print(val_labels[1].shape)
                    sample = np.array(samples)
                    print(sample.shape)
                    for random_index in range(16):
                        generate_image = samples[-1][random_index].reshape(in_channels, image_size, image_size)
                        plt.hist(val_features[1].cpu().numpy().flatten(), color='red', range=[-1, 1], edgecolor='black',
                            label='noise',bins=20)
                        plt.hist(generate_image.flatten(), color='blue', range=[-1, 1], edgecolor='black', label='pred', bins=20)
                        plt.legend()
                        plt.xlabel('pressure')
                        plt.ylabel('Frequency')
                    # plt.title('Histogram with Legend')
                        plt.savefig(f'sample\\epoch{i}group{val_steps}random{random_index}.jpg', format='jpg', dpi=200)
                        plt.close('all')
                        print('data_saved')
            '''
            if 'model_save_path' in kwargs.keys() and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_best_model(epoch=i, model=model, optimizer=optimizer, path=kwargs['model_save_path'])
            '''
        tensorboard_writer.close()
        return model





# optim = torch.optim.Adam(DDPM_Trainer.parameters(), lr=0.001)
optimizer = Adadelta(DDPM.parameters(), lr=0.001, weight_decay=0.001)
epoches = 8000
root_path = "./Model_save_package"
setting = "Adadelta_imageSize{}_channels{}_timeSteps{}_epoch{}_scheduleName{}".format(image_size, in_channels, len(betas.shape), epoches, schedule_name)

saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

# 断点续训
checkpoint_path = saved_path +'/'+'BestModel.pth'
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    DDPM.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
# 断后继续训练的epoch数
num_epochs = 8000
Trainer = DDPM_Trainer(start=start_epoch, val_loader=val_loader, IFEarlyStopping=True, patience=8,
                       epochs=num_epochs, train_loader=train_loader,
                       optimizer=optimizer, device=device,
                       IFadjust_learning_rate=True, types='type1')


DDPM = Trainer(DDPM,  model_save_path=saved_path)
