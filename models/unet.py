from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "get_timestep_embedding",
    "Upsample",
    "Downsample",
    "ResnetBlock",
    "AttnBlock",
    "DiffusionUNet",
]


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Build sinusoidal timestep embeddings."""
    if timesteps.ndim != 1:
        raise ValueError(f"Expected a 1D timestep tensor, but got shape {timesteps.shape}.")

    half_dim = embedding_dim // 2
    if half_dim < 1:
        raise ValueError(f"Embedding dimension must be >= 2, but got {embedding_dim}.")

    exponent = math.log(10000) / max(half_dim - 1, 1)
    frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -exponent)
    args = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def normalize(in_channels: int) -> nn.GroupNorm:
    num_groups = min(32, in_channels)
    while in_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1) if with_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.conv is not None else x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0) if with_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_shortcut = None
        self.nin_shortcut = None
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))

        if self.in_channels != self.out_channels:
            if self.conv_shortcut is not None:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, height, width = q.shape
        q = q.reshape(b, 1, c, height * width).transpose(-2, -1)
        k = k.reshape(b, 1, c, height * width).transpose(-2, -1)
        v = v.reshape(b, 1, c, height * width).transpose(-2, -1)

        if self.use_sdpa:
            h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            weights = torch.matmul(q, k.transpose(-1, -2)) * (c ** -0.5)
            weights = torch.softmax(weights, dim=-1)
            h = torch.matmul(weights, v)

        h = h.transpose(-2, -1).reshape(b, c, height, width)
        h = self.proj_out(h)
        return x + h


class DiffusionUNet(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        ch = config.model.ch
        out_ch = config.model.out_ch
        ch_mult = tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = set(config.model.attn_resolutions)
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != x.shape[3] or x.shape[2] != self.resolution:
            raise ValueError(
                f"DiffusionUNet expects square patches of size {self.resolution}, "
                f"but received shape {tuple(x.shape)}."
            )

        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)