import torch as th
import torch.nn as nn
import torch.nn.functional as F

from Model.Model_utils import timestep_embedding, LayerNorm, Transformer, convert_module_to_f16
from Model.Unet import UNetModel



class Text2ImUNet(UNetModel):
    """
    一个条件在输入压力点的编码上的UnetModel。
    需要一个额外的kwarg，p_emb

    :param text_ctx: 预期的 text token 数量         number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: 在输出层之后使用 LayerNorm.
    :param tokenizer: 文本标记器的取样/词汇表大小.
    """

    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        input_embedding_seqnum,
        *args,
        cache_text_emb=False,
        xf_ar=0.0,
        xf_padding=False,
        share_unemb=False,
        **kwargs,
    ):
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        self.input_embedding_seqnum = input_embedding_seqnum

        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            super().__init__(*args, **kwargs, encoder_channels=xf_width)
        if self.xf_width:
            self.transformer = Transformer(
                text_ctx,
                xf_width,
                xf_layers,
                xf_heads,
            )
            if xf_final_ln:
                self.final_ln = LayerNorm(xf_width)
            else:
                self.final_ln = None

            self.token_embedding = nn.Linear(self.input_embedding_seqnum, xf_width)
            nn.init.xavier_uniform_(self.token_embedding.weight)
            nn.init.zeros_(self.token_embedding.bias)
            self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32))
            self.transformer_proj = nn.Linear(xf_width, self.model_channels * 4)

            if self.xf_padding:
                self.padding_embedding = nn.Parameter(
                    th.empty(text_ctx, xf_width, dtype=th.float32)
                )
            if self.xf_ar:
                self.unemb = nn.Linear(xf_width, self.input_embedding_seqnum)
                if share_unemb:
                    self.unemb.weight = self.token_embedding.weight

        self.cache_text_emb = cache_text_emb
        self.cache = None

    def convert_to_fp16(self):
        super().convert_to_fp16()
        if self.xf_width:
            self.transformer.apply(convert_module_to_f16)
            self.transformer_proj.to(th.float16)
            self.token_embedding.to(th.float16)
            self.positional_embedding.to(th.float16)
            if self.xf_padding:
                self.padding_embedding.to(th.float16)
            if self.xf_ar:
                self.unemb.to(th.float16)

    def get_text_emb(self, tokens):
        assert tokens is not None
        '''
        if self.cache_text_emb and self.cache is not None:
            assert (
                tokens == self.cache["tokens"]
            ).all(), f"Tokens {tokens.cpu().numpy().tolist()} do not match cache {self.cache['tokens'].cpu().numpy().tolist()}"
            return self.cache
        '''


        xf_in = self.token_embedding(tokens)

        xf_in = xf_in + self.positional_embedding[None]

        xf_out = self.transformer(xf_in.to(self.dtype))

        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)
        # print(outputs)
        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(),
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, tokens=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.xf_width:
            text_outputs = self.get_text_emb(tokens)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        h = x.type(self.dtype)  # 转换数据类型float16或float32
        i=1
        for module in self.input_blocks:
            # print('i{}h{},emb{}xf_out{}'.format(i,h.shape,emb.shape,xf_out.shape) )
            h = module(h, emb, xf_out)
            hs.append(h)
            i+=1
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)        # pop删除列表中最后一个
            h = module(h, emb, xf_out)
            # print('out_h', h.shape)
        h = h.type(x.dtype)
        h = self.out(h)
        return h


class SuperResText2ImUNet(Text2ImUNet):
    """
    A text2im model that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class InpaintText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )


class SuperResInpaintText2ImUnet(Text2ImUNet):
    """
    A text2im model which can perform both upsampling and inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 3 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 3 + 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        timesteps,
        inpaint_image=None,
        inpaint_mask=None,
        low_res=None,
        **kwargs,
    ):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )
