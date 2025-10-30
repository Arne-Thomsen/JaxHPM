import jax
import jax.numpy as jnp

from flax import nnx
from tqdm import tqdm

from jaxhpm.splines import NeuralSplineFourierFilterNNX
from jaxhpm.painting import cic_paint, cic_read
from jaxhpm.utils import BaseModel


def batched_eval(model, in_array, batch_size):
    assert in_array.ndim == 2

    preds = []
    for i in tqdm(range(in_array.shape[0] // batch_size)):
        preds.append(model(in_array[i * batch_size : (i + 1) * batch_size]))
    preds.append(model(in_array[(i + 1) * batch_size :]))

    return jnp.concatenate(preds, axis=0)


class MLP(BaseModel):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        n_hidden: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.0,
        activation=jax.nn.relu,
        norm_type: str = "layer",
        k_filter: str = None,
    ):
        super().__init__()

        self.linear_in = nnx.Linear(d_in, d_hidden, rngs=rngs)
        self.linear_hid = [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]
        self.linear_out = nnx.Linear(d_hidden, d_out, rngs=rngs)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type

        if k_filter == "gaussian":
            self.k_smooth = nnx.Param(jnp.array(1.0))
        elif k_filter == "spline":
            self.k_model = NeuralSplineFourierFilterNNX(n_knots=8, d_latent=16, rngs=rngs)
        elif k_filter is not None:
            raise ValueError(f"Unsupported k_filter: {k_filter}")

        if isinstance(self.activation, str):
            if self.activation == "relu":
                self.activation = jax.nn.relu
            elif self.activation == "swish":
                self.activation = jax.nn.swish
            elif self.activation == "sigmoid":
                self.activation = jax.nn.sigmoid
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")

        if self.dropout_rate > 0:
            self.dropout = [nnx.Dropout(dropout_rate, rngs=rngs) for _ in range(n_hidden)]

        if self.norm_type == "layer":
            self.norm_in = nnx.LayerNorm(d_hidden, rngs=rngs)
            self.norm_hid = [nnx.LayerNorm(d_hidden, rngs=rngs) for _ in range(n_hidden)]
        elif self.norm_type == "batch":
            self.norm_in = nnx.BatchNorm(d_hidden, rngs=rngs)
            self.norm_hid = [nnx.BatchNorm(d_hidden, rngs=rngs) for _ in range(n_hidden)]

        self.d_out = d_out

    def __call__(self, x, training: bool = False):
        x = self.linear_in(x)

        if self.norm_type == "layer":
            x = self.norm_in(x)
        elif self.norm_type == "batch":
            x = self.norm_in(x, use_running_average=not training)

        x = self.activation(x)

        for i, linear in enumerate(self.linear_hid):
            x = linear(x)

            if self.norm_type == "layer":
                x = self.norm_hid[i](x)
            elif self.norm_type == "batch":
                x = self.norm_hid[i](x, use_running_average=not training)

            x = self.activation(x)
            if training and self.dropout_rate > 0:
                x = self.dropout[i](x, deterministic=not training)

        x = self.linear_out(x)
        return x


class ConditionedCNN(BaseModel):
    def __init__(
        self,
        d_in=4,
        d_hidden=64,
        d_out=1,
        n_hidden=4,
        kernel_size=(3, 3, 3),
        activation=jax.nn.swish,
        use_residual=True,
        norm_type="layer",
        group_norm_groups: int | None = None,
        rngs=nnx.Rngs(0),
    ):
        super().__init__()

        self.activation = activation
        self.use_residual = use_residual
        self.norm_type = norm_type
        self.d_out = d_out

        # CNN
        self.conv_in = nnx.Conv(d_in, d_hidden, kernel_size, padding="CIRCULAR", rngs=rngs)
        self.conv_hidden = [
            nnx.Conv(d_hidden, d_hidden, kernel_size, padding="CIRCULAR", rngs=rngs) for _ in range(n_hidden)
        ]
        self.conv_out = nnx.Conv(d_hidden, d_out, kernel_size, padding="CIRCULAR", rngs=rngs)

        if self.norm_type == "layer":
            self.norm_in = nnx.LayerNorm(d_hidden, rngs=rngs)
            self.norm_hidden = [nnx.LayerNorm(d_hidden, rngs=rngs) for _ in range(n_hidden)]
        elif self.norm_type == "batch":
            self.norm_in = nnx.BatchNorm(d_hidden, rngs=rngs)
            self.norm_hidden = [nnx.BatchNorm(d_hidden, rngs=rngs) for _ in range(n_hidden)]
        elif self.norm_type == "group":
            # Choose a reasonable default number of groups for 3D CNNs
            def _pick_groups(c: int, desired: int = 32) -> int:
                g = min(desired, c)
                # Ensure divisibility
                while c % g != 0 and g > 1:
                    g -= 1
                return g

            gn_groups = group_norm_groups or _pick_groups(d_hidden)
            # Normalize over spatial dims only (channels-last), excluding batch and channel
            # Works for both (H, W, D, C) and (B, H, W, D, C)
            reduction_axes = (-4, -3, -2)
            self.norm_in = nnx.GroupNorm(d_hidden, num_groups=gn_groups, reduction_axes=reduction_axes, rngs=rngs)
            self.norm_hidden = [
                nnx.GroupNorm(d_hidden, num_groups=gn_groups, reduction_axes=reduction_axes, rngs=rngs)
                for _ in range(n_hidden)
            ]

        # scale conditioning https://arxiv.org/abs/1709.07871
        self.scale_embed = nnx.Linear(1, d_hidden, rngs=rngs)
        self.film_gamma = [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]
        self.film_beta = [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]

    def __call__(self, x, scale, training=False):
        """Forward pass through the conditioned CNN.

        Args:
            x: Input tensor of shape (mesh, mesh, mesh, features) or (batch, mesh, mesh, mesh, features)
            scale: Scale factor(s), shape () for single scale or (batch,) for batched scales
            training: Whether in training mode

        Returns:
            Output tensor of same spatial shape as input
        """
        # Determine if input is batched
        is_batched = x.ndim == 5  # (batch, H, W, D, features) vs (H, W, D, features)

        x = self.conv_in(x)

        if self.norm_type == "layer":
            x = self.norm_in(x)
        elif self.norm_type == "batch":
            x = self.norm_in(x, use_running_average=not training)
        elif self.norm_type == "group":
            x = self.norm_in(x)

        x = self.activation(x)

        # Handle scale embedding for both batched and non-batched inputs
        if is_batched:
            # Ensure scale has batch dimension
            if scale.ndim == 0:  # Single scale for all batch items
                scale = jnp.broadcast_to(scale, (x.shape[0],))
            elif scale.ndim == 1 and scale.shape[0] != x.shape[0]:
                raise ValueError(f"Scale batch size {scale.shape[0]} doesn't match input batch size {x.shape[0]}")

            scale_embedding = self.activation(self.scale_embed(scale.reshape(-1, 1)))
        else:
            # Non-batched case
            scale_embedding = self.activation(self.scale_embed(scale.reshape(1, 1)))

        # Hidden layers with scale conditioning
        for i, conv in enumerate(self.conv_hidden):
            if self.use_residual:
                residual = x

            x = conv(x)

            if self.norm_type == "layer":
                x = self.norm_hidden[i](x)
            elif self.norm_type == "batch":
                x = self.norm_hidden[i](x, use_running_average=not training)
            elif self.norm_type == "group":
                x = self.norm_hidden[i](x)

            # Apply FiLM conditioning (scale and shift based on scale factor)
            gamma = self.film_gamma[i](scale_embedding)
            beta = self.film_beta[i](scale_embedding)

            # Reshape for broadcasting based on input dimensionality
            if is_batched:
                # Shape: (batch, 1, 1, 1, d_hidden)
                gamma = gamma.reshape(gamma.shape[0], 1, 1, 1, -1)
                beta = beta.reshape(beta.shape[0], 1, 1, 1, -1)
            else:
                # Shape: (1, 1, 1, d_hidden)
                gamma = gamma.reshape(1, 1, 1, -1)
                beta = beta.reshape(1, 1, 1, -1)

            # Apply conditioning
            x = x * gamma + beta

            if self.use_residual:
                x = x + residual

            x = self.activation(x)

        x = self.conv_out(x)
        return x
