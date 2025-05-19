import jax.numpy as jnp
from jax.experimental.sparse import BCOO


def build_sparse_conv_matrix(kernel: jnp.ndarray, kernel_window: int) -> BCOO:
    """
    Builds a sparse convolution matrix (valid mode) using JAX.
    Args:
        kernel: the kernel vector.
        kernel_window: the kernel window.

    Returns:
        the sparse convolution matrix.
    """
    n = len(kernel)
    out_dim = n - kernel_window + 1

    # Flatten kernel to ensure it's 1D
    kernel_flat = kernel[:kernel_window].reshape(-1)

    row_idx = jnp.repeat(jnp.arange(out_dim), kernel_window)
    col_idx = jnp.tile(jnp.arange(kernel_window), out_dim) + row_idx
    data = jnp.tile(kernel_flat, out_dim)

    coords = jnp.stack([row_idx, col_idx], axis=1)

    return BCOO((data, coords), shape=(out_dim, n))
