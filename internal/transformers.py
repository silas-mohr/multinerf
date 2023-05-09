# -----------------------------------------------------------
#
#    JAX implementation of a LearnedEmbeddings transformer
#    Inspired by ABLE-NeRF
#
# -----------------------------------------------------------

import functools
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from internal import math
import gin


class EncoderBlock(nn.Module):
    weight_init: str = "he_uniform"
    input_dim: int = 128  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads: int = 8  # Number of heads in multi headed attention
    ff_ratio: int = 2  # Feed forward ratio
    dropout_prob: float = 0.0  # Dropout probability
    train: bool = True

    def setup(self):
        self.dim_feedforward = self.input_dim * self.ff_ratio

    @nn.compact
    def __call__(self, x, kv, mask=None):
        if len(kv.shape) != len(x.shape):
            x = jnp.repeat(x[None, None, ...], kv.shape[0], axis=0)
        dense_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
        )
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_prob,
        )(x, kv, mask=mask)
        x = nn.LayerNorm()(x + attn_out)

        linear_out = nn.Sequential(
            [
                dense_layer(self.dim_feedforward),
                nn.gelu,
                nn.Dropout(rate=self.dropout_prob, deterministic=not self.train),
                dense_layer(self.input_dim),
            ]
        )(x)
        linear_out = nn.Dropout(self.dropout_prob, deterministic=not self.train)(
            linear_out
        )
        x = nn.LayerNorm()(linear_out + x)

        return x


class LETransformer(nn.Module):
    dim: int = 128
    ff_ratio: int = 2
    dropout: float = 0.2
    lp_atten_layer: int = 2

    def setup(self):
        self.num_heads = self.dim // 32

    @nn.compact
    def __call__(self, tokens, light_probes, class_token) -> Any:
        x = EncoderBlock(
            input_dim=self.dim,
            num_heads=self.num_heads,
            ff_ratio=self.ff_ratio,
            dropout_prob=self.dropout,
        )(light_probes, tokens)

        for _ in range(self.lp_atten_layer):
            x = EncoderBlock(
                input_dim=self.dim,
                num_heads=self.num_heads,
                ff_ratio=self.ff_ratio,
                dropout_prob=self.dropout,
            )(x, x)

        view_token = jnp.repeat(
            class_token[..., None, :], 128, axis=len(tokens.shape) - 2
        )
        x = EncoderBlock(
            input_dim=self.dim,
            num_heads=self.num_heads,
            ff_ratio=self.ff_ratio,
            dropout_prob=self.dropout,
        )(view_token, x)

        return x


class Embedding(nn.Module):
    in_channels: int = 3
    N_freqs: int = 4
    logscale: bool = False

    def setup(self):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        self.funcs = [math.safe_sin, math.safe_cos]
        self.out_channels = self.in_channels * (len(self.funcs) * self.N_freqs + 1)

        if self.logscale:
            self.freq_bands = 2 ** jnp.linspace(0, self.N_freqs - 1, self.N_freqs)
        else:
            self.freq_bands = jnp.linspace(1, 2 ** (self.N_freqs - 1), self.N_freqs)

    def __call__(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return jnp.concatenate(out, -1)


@gin.configurable
class LearnedEmbeddings(nn.Module):
    weight_init: str = "he_uniform"
    num_light_probes: int = 100
    lp_dim: int = 256
    ff_ratio: int = 2
    dropout: float = 0.0
    lp_atten_layer: int = 2
    dir_L_bands: int = 4

    def setup(self):
        self.light_probes = self.param(
            "LE",
            getattr(jax.nn.initializers, self.weight_init)(),
            (1, self.num_light_probes, self.lp_dim),
        )
        self.in_channels_dir = self.dir_L_bands * 3 * 2 + 3

    @nn.compact
    def __call__(self, tokens, viewdirs):
        dense_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
        )
        view_token = Embedding(3, self.in_channels_dir)(viewdirs)
        view_token = dense_layer(self.lp_dim)(view_token)
        spec_color = LETransformer(
            dim=self.lp_dim,
            ff_ratio=self.ff_ratio,
            dropout=self.dropout,
            lp_atten_layer=self.lp_atten_layer,
        )(
            tokens=tokens,
            light_probes=self.light_probes,
            class_token=view_token,
        )

        return spec_color
