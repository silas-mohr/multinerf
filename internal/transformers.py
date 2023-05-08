# -----------------------------------------------------------
#
#    JAX implementation of ABLE-NeRF LE transformer
#
# -----------------------------------------------------------

from typing import Any
import jax
import jax.numpy as jnp
from flax import linen as nn
import gin


class EncoderBlock(nn.Module):
    input_dim: int = 256  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads: int = 8  # input_dim=256, num_heads=8, ff_ratio=3, dropout_p=0.0
    ff_ratio: int = 2  # Feed forward ratio
    dropout_prob: float = 0.0  # Dropout probability

    def setup(self):
        self.dim_feedforward = self.input_dim * self.ff_ratio

    @nn.compact
    def __call__(self, x, mask=None, train=True):
        attn_out, weights = self.self_attn = nn.MultiHeadDotProductAttention(
            embed_dim=self.input_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_prob,
        )(x, mask=mask)
        x = nn.LayerNorm()(x + attn_out)

        linear_out = nn.Sequential(
            nn.Dense(self.dim_feedforward),
            nn.gelu,
            nn.Dropout(self.dropout_prob, deterministic=not train),
            nn.Dense(self.input_dim),
        )(x)
        linear_out = nn.Dropout(self.dropout_prob)(linear_out)
        x = nn.LayerNorm()(linear_out + x)

        return x, weights


class LETransformer(nn.Module):
    dim: int = 256
    ff_ratio: int = 2
    dropout: float = 0.2
    lp_atten_layer: int = 2

    def setup(self):
        self.num_heads = self.dim // 64

    @nn.compact
    def __call__(self, tokens, light_probes, class_token) -> Any:
        x, _ = EncoderBlock(
            input_dim=self.dim,
            num_heads=self.num_heads,
            ff_ratio=self.ff_ratio,
            dropout_p=self.dropout,
        )(light_probes, tokens)

        for _ in range(self.lp_atten_layer):
            x, _ = EncoderBlock(
                input_dim=self.dim,
                num_heads=self.num_heads,
                ff_ratio=self.ff_ratio,
                dropout_p=self.dropout,
            )(x, x)

        x, _ = EncoderBlock(
            input_dim=self.dim,
            num_heads=self.num_heads,
            ff_ratio=self.ff_ratio,
            dropout_p=self.dropout,
        )(class_token.unsqueeze(1), x)

        return x


@gin.configurable
class LearnedEmbeddings(nn.Module):
    weight_init: str = "he_uniform"
    num_light_probes: int = 100
    lp_dim: int = 256
    ff_ratio: int = 2
    dropout: float = 0.0
    lp_atten_layer: int = 2

    def setup(self):
        self.light_probes = self.param(
            "LE",
            getattr(jax.nn.initializers, self.weight_init)(),
            jnp.randn(1, self.num_light_probes, self.lp_dim),
        )

        dir_L_bands = 4
        # self.fourier_dir_emb = Embedding(3, dir_L_bands)
        self.in_channels_dir = dir_L_bands * 3 * 2 + 3

    def __call__(self, tokens):
        spec_color = LETransformer(
            dim=self.lp_dim,
            ff_ratio=self.ff_ratio,
            dropout=self.dropout,
            lp_atten_layer=self.lp_atten_layer,
        )(tokens=tokens, light_probes=self.light_probes, class_token=class_token,)
        spec_color = nn.Sequential(
            nn.Dense(self.lp_dim), nn.relu, nn.Dense(3), nn.sigmoid
        )(spec_color)
