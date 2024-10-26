import torch
from Model.Network_Helper import *
from Model.varianceSchedule import VarianceSchedule
from torch import nn, optim
from torch.nn import functional as F
class DiffusionModel(nn.Module):
    def __init__(self, schedule_name='linear_beta_schedule', timesteps=1000, beta_start=0.0001, beta_end=0.02, denoise_model=None):
        super(DiffusionModel, self).__init__()
        self.denoise_model = denoise_model

        # 方差生成
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)     # 与前面所有项累乘的数组
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)       # F.pad函数向数组左边补一个1, 右边补0个1
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 计算扩散q(x_t | x_{t-1})与其它
        # x_t = sqrt(alphas_cumprod) * x_0 + sqrt(1 - alphas_cumprod) * z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 计算后验q(x_{x_t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    # 第时间步t加噪声的过程
    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        # x_t = sqrt(alphas_cumprod) * x_0 + sqrt(1 - alphas_cumprod) * z_t
        if noise is None:
            noise = torch.randn_like(x_start)       # 高斯噪声

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # 计算loss
    def compute_loss(self, x_start, t, noise=None, label=None, loss_type='l1'):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t, label)

        predicted_noise.retain_grad()
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)        # MAEloss
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.huber_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss, noise, predicted_noise

    def evaluate_loss(self, x_start, t, noise=None, loss_type='l1', label=None):
        if noise is None:
            noise = torch.randn_like(x_start, requires_grad=True)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print('x_noisy', x_noisy.shape)
        predicted_noise = self.denoise_model(x_noisy, t, label)
        # print('predicted_noise', predicted_noise.shape)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)        # MAEloss
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
            #if (noise - predicted_noise).max().item()>5:
                # print((noise - predicted_noise).max().item(), noise.max().item(), predicted_noise.max().item())
                # print(noise)
        elif loss_type == 'huber':
            loss = F.huber_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss, noise, predicted_noise
    @torch.no_grad()    # 不需要计算梯度
    def p_sample(self, x, t, label, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # equation 11 in the paper
        # use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.denoise_model(x, t, label) / sqrt_one_minus_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()  # 不需要计算梯度
    def p_sample_loop(self, shape, label):
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        # start from pure noise(for each example in the batch) 纯噪声
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), label, i)
            imgs.append(img.cpu().numpy())

        return imgs

    @torch.no_grad()  # 不需要计算梯度
    def sample(self, labels, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size), label=labels)

    def forward(self, mode, **kwargs):
        if mode == 'train':
            # 必须先判断参数
            if 'x_start' and 't' in kwargs.keys():
                if 'loss_type' and 'noise' in kwargs.keys():
                    return self.compute_loss(x_start=kwargs['x_start'], t=kwargs['t'], noise=kwargs['noise'], loss_type=kwargs['loss_type'], label=kwargs['label'])
                elif 'loss_type' in kwargs.keys():
                    return self.compute_loss(x_start=kwargs['x_start'], t=kwargs['t'], loss_type=kwargs['loss_type'], label=kwargs['label'])
                elif 'noise' in kwargs.keys():
                    return self.compute_loss(x_start=kwargs['x_start'], t=kwargs['t'], noise=kwargs['noise'], label=kwargs['label'])
                else:
                    return self.compute_loss(x_start=kwargs['x_start'], t=kwargs['t'], label=kwargs['label'])
            else:
                raise ValueError('扩散模型在训练时必须传入参数x_start和t!')

        elif mode == 'eval':
            # 验证阶段计算损失，但不会涉及retain_grad
            return self.evaluate_loss(x_start=kwargs['x_start'], t=kwargs['t'], loss_type=kwargs['loss_type'],
                                      label=kwargs['label'])
        elif mode == 'generate':
            if 'image_size' and 'batch_size' and 'channels' in kwargs.keys():
                return self.sample(image_size=kwargs['image_size'],
                                   batch_size=kwargs['batch_size'],
                                   channels=kwargs['channels'],
                                   labels=kwargs['labels'])
            else:
                raise ValueError('扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数')

        else:
            raise ValueError('mode参数必须从{train}和{generate}两种模式中选择')


