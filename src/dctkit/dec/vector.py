import jax.numpy as jnp
import numpy.typing as npt
from .cochain import CochainD1
from dctkit.mesh import simplex as spx
from jax import Array


class DiscreteTensorField():
    """Discrete tensor fields class.

    Args:
        S: the simplicial complex where the discrete vector field is defined.
        is_primal: True if the discrete vector field is primal, False otherwise.
        coeffs: array of the coefficients of the discrete vector fields.
        rank: rank of the tensor.
    """

    def __init__(self, S: spx.SimplicialComplex, is_primal: bool,
                 coeffs: npt.NDArray | Array, rank: int):
        self.S = S
        self.is_primal = is_primal
        self.coeffs = coeffs
        self.rank = rank


class DiscreteVectorField(DiscreteTensorField):
    """Inherited class for discrete vector fields."""

    def __init__(self, S: spx.SimplicialComplex, is_primal: bool,
                 coeffs: npt.NDArray | Array):
        super().__init__(S, is_primal, coeffs, 1)


class DiscreteVectorFieldD(DiscreteVectorField):
    """Inherited class for dual discrete vector fields."""

    def __init__(self, S: spx.SimplicialComplex, coeffs: npt.NDArray | Array):
        super().__init__(S, False, coeffs)


class DiscreteTensorFieldD(DiscreteTensorField):
    """Inherited class for dual discrete tensor fields."""

    def __init__(self, S: spx.SimplicialComplex, coeffs: npt.NDArray | Array,
                 rank: int):
        super().__init__(S, False, coeffs, rank)


def flat_DPD(v: DiscreteTensorFieldD) -> CochainD1:
    """Implements the flat operator for dual discrete vector fields.

    Args:
        v: a dual discrete vector field.
    Returns:
        the dual cochain resulting from the application of the flat operator.
    """
    dedges = v.S.dual_edges_vectors[:, :v.coeffs.shape[0]]
    flat_matrix = v.S.flat_weights
    # multiply weights of each dual edge by the vectors associated to the dual nodes
    # belonging to the edge
    weighted_v = v.coeffs @ flat_matrix
    weighted_v_T = weighted_v.T
    if v.rank == 1:
        # vector field case
        # perform dot product row-wise with the edge vectors
        # of the dual edges (see definition of DPD in Hirani, pag. 54).
        coch_coeffs = jnp.einsum("ij, ij -> i", weighted_v_T, dedges)
    elif v.rank == 2:
        # tensor field case
        # apply each matrix (rows of the multiarray weighted_v_T fixing the first axis)
        # to the edge vector of the corresponding dual edge
        coch_coeffs = jnp.einsum("ijk, ik -> ij", weighted_v_T, dedges)
    return CochainD1(v.S, coch_coeffs)