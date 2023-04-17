import numpy as np
import dctkit as dt
import jax.numpy as jnp
from jax import grad, Array
from dctkit.math.opt import optctrl
from scipy import sparse
import matplotlib.pyplot as plt
import os
from dctkit.apps import elastica as el
import numpy.typing as npt
import pytest


@pytest.mark.parametrize('tune_EI0', [True, False])
def test_elastica(setup_test, tune_EI0):
    data = "xy_F_20.txt"
    F = -20
    np.random.seed(42)

    # load true data
    filename = os.path.join(os.path.dirname(__file__), data)
    data = np.genfromtxt(filename)

    # sample true data
    sampling = 10

    density = 1
    num_elements_data = int(100/sampling)
    num_elements = num_elements_data*density

    R_0 = 1e-2
    R_1 = 1e-2
    L = 1
    h = L/(num_elements)
    rho = R_1/R_0

    ela = el.ElasticaProblem(num_elements=num_elements, L=L, rho=rho)
    num_nodes = ela.S.num_nodes

    # define I_0
    I_0 = np.pi/4 * R_0**4

    # bidiagonal matrix to transform theta in (x,y)
    diag = [1]*(num_nodes)
    upper_diag = [-1]*(num_nodes-1)
    upper_diag[0] = 0
    diags = [diag, upper_diag]
    transform = sparse.diags(diags, [0, -1]).toarray()
    transform[1, 0] = -1

    # initial guess for the angles (EXCEPT THE FIRST ANGLE, FIXED BY BC)
    theta_0 = 0.1*np.random.rand(num_elements-1).astype(dt.float_dtype)

    # compute true solution and add noise
    noise = 0.0*np.random.rand(num_elements_data+1)
    x_true = data[:, 1][::sampling] + noise
    y_true = data[:, 2][::sampling] + noise
    theta_true = np.empty(num_elements_data, dtype=dt.float_dtype)
    for i in range(num_elements_data):
        theta_true[i] = np.arctan(
            (y_true[i+1]-y_true[i])/(x_true[i+1]-x_true[i]))

    # state function: stationarity conditions of the elastic energy
    def statefun(x: npt.NDArray, theta_0: npt.NDArray, F: float) -> Array:
        u = x[:-1]
        EI0 = x[-1]
        return grad(ela.energy_elastica)(u, EI0, theta_0, F)

    if tune_EI0:

        # define extra_args
        constraint_args = {'theta_0': theta_true[0], 'F': F}
        obj_args = {'theta_true': theta_true}

        def obj(x: npt.NDArray, theta_true: npt.NDArray) -> Array:
            theta_guess = x[:-1]
            EI_guess = x[-1]
            return ela.obj_fun_theta(theta_guess, EI_guess, theta_true)

        prb = optctrl.OptimalControlProblem(objfun=obj,
                                            statefun=statefun,
                                            state_dim=num_elements-1,
                                            nparams=num_elements,
                                            constraint_args=constraint_args,
                                            obj_args=obj_args)
        EI0_0 = 1*np.ones(1, dtype=dt.float_dtype)
        x0 = np.concatenate((theta_0, EI0_0))
        x = prb.run(x0=x0)
        theta = x[:-1]
        EI0 = x[-1]
        # extend theta
        theta = np.insert(theta, 0, theta_true[0])
        print(f"The optimal E*I_0 is {EI0}")
        print(f"The optimal E is {EI0/I_0}")
    else:
        args = {'EI0': 7.8489, 'theta_0': theta_true[0], 'F': F}

        def obj(x: npt.NDArray, EI0: float, theta_0: npt.NDArray, F: float) -> float:
            return ela.energy_elastica(x, EI0, theta_0, F)

        prb = optctrl.OptimizationProblem(
            dim=num_elements-1, state_dim=num_elements-1, objfun=obj)

        prb.set_obj_args(args)
        theta = prb.run(x0=theta_0)

        theta = np.insert(theta, 0, theta_true[0])

    # plot theta
    # plt.plot(theta)
    # plt.show()

    # reconstruct x, y
    cos_theta = h*jnp.cos(theta)
    sin_theta = h*jnp.sin(theta)
    b_x = jnp.insert(cos_theta, 0, 0)
    b_y = jnp.insert(sin_theta, 0, 0)
    x = jnp.linalg.solve(transform, b_x)
    y = jnp.linalg.solve(transform, b_y)
    x = x[::density]
    y = y[::density]

    # plot the results
    # plt.plot(x_true, y_true, 'r')
    # plt.plot(x, y, 'b')
    # plt.show()
    error = np.linalg.norm(x - x_true) + np.linalg.norm(y - y_true)
    assert error <= 2e-2
