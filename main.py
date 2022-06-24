import tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pinn import PINN
from network import Network
from optimizer import L_BFGS_B



def mass_cons(network, xy):
    """
    Compute the components of the mass conseervation (u_x, v_y)
    Args:
        network
        xy: network input variables as ndarray.
    Returns:
        (u_x, v_y) as ndarray.
    """

    xy = tf.constant(xy)
    with tf.GradientTape() as g:
        g.watch(xy)
        uv = network(xy)
    uv_j = g.batch_jacobian(uv, xy)
    u_x =  uv_j[..., 0, 0]
    v_y =  uv_j[..., 0, 1]

    return u_x.numpy(), v_y.numpy()


def u_0(xy):
    """
    Initial wave form.
    Args:
        tx: variables (t, x) as tf.Tensor.
    Returns:
        u(t, x) as tf.Tensor.
    """

    x = xy[..., 0, None]
    y = xy[..., 1, None]


    return    4*y*(1 - y) 


if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model
    for the cavity flow governed by the steady Navier-Stokes equation.
    """

    # number of training samples
    num_train_samples = 5000
    # number of test samples
    num_test_samples = 200

    # inlet flow velocity
    # density
    rho = 1
    # viscosity
    mu = 1e-2
    # Re = rho/mu

    # build a core network model
    network = Network().build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, rho=rho, mu=mu).build()

    # circle
    x_f =2
    x_ini=0
    y_f=2
    y_ini=0

    # create training input
    xyt_eqn = np.random.rand(num_train_samples, 2)
    xyt_eqn[...,0] = (x_f - x_ini)*xyt_eqn[...,0] + x_ini
    xyt_eqn[...,1] = (y_f - y_ini)*xyt_eqn[...,1] + y_ini

    for i in range(num_train_samples):
      while xyt_eqn[i, 0] < 1 and xyt_eqn[i, 1] > 1 :
        xyt_eqn[i, 0] = (x_f - x_ini) * np.random.rand(1, 1) + x_ini
        xyt_eqn[i, 1] = (y_f - y_ini) * np.random.rand(1, 1) + y_ini

    xyt_w1 = np.random.rand(num_train_samples, 2)  # wall 1
    xyt_w1[..., 0] = (x_f - x_ini)*xyt_w1[...,0] + x_ini
    xyt_w1[..., 1] =  0       

    xyt_w2 = np.random.rand(num_train_samples, 2)  # wall 2
    xyt_w2[..., 0] = 2
    xyt_w2[..., 1] =  (y_f - y_ini)*xyt_w2[...,1] + y_ini

    xyt_w3 = np.random.rand(num_train_samples, 2)  # wall 3
    xyt_w3[..., 0] = 1
    xyt_w3[..., 1] = (y_f - 1)*xyt_w3[...,1] + 1

    xyt_w4 = np.random.rand(num_train_samples, 2)  # wall 4
    xyt_w4[..., 0] = (1 - x_ini)*xyt_w4[...,0] + x_ini
    xyt_w4[..., 1] = 1

    xyt_out = np.random.rand(num_train_samples, 2)  # output
    xyt_out[..., 0] = xyt_out[...,0] + 1
    xyt_out[..., 1] =  2 

    xyt_in = np.random.rand(num_train_samples, 2) # input
    xyt_in[...,0] = 0

    x_train = [xyt_eqn, xyt_w1, xyt_w2, xyt_w3, xyt_w4, xyt_out, xyt_in]

    # create training output
    zeros = np.zeros((num_train_samples, 3))
    a = u_0(tf.constant(xyt_in)).numpy()
    b = np.zeros((num_train_samples, 1))
    onze = np.random.permutation(np.concatenate([a,b,a],axis = -1))
    y_train = [zeros, onze, zeros, zeros, zeros, zeros, zeros]

    # train the model using L-BFGS-B algorithm
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()

    # create meshgrid coordinates (x, y) for test plots     

    x = np.linspace(x_ini, x_f, num_test_samples)
    y = np.linspace(y_ini, y_f, num_test_samples)
    x, y = np.meshgrid(x, y)
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)
    # predict (u,v,p)
    u_v_p = network.predict(xy, batch_size=len(xy))
    u, v, p = [ u_v_p[..., i].reshape(x.shape) for i in range(u_v_p.shape[-1]) ]
    # compute (u, v)
    u = u.reshape(x.shape)
    v = v.reshape(x.shape)
    p = p.reshape(x.shape)
    
    # plot test results
    
    ########################### Pressure
    from matplotlib.patches import Rectangle
    font1 = {'family':'serif','size':40}

    fig0, ax0 = plt.subplots(1, 1,figsize=(12,10))
    cf0 = ax0.contourf(x, y, p, np.arange(-0.2, 1, .05),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, pad=0.03, aspect=25, format='%.0e')
    plt.title("p", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Rectangle((0, 1), 1, 1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar0.ax.tick_params(labelsize=35)
    plt.show()

    ########################### X-velocity

    fig0, ax0 = plt.subplots(1, 1, figsize=(12,10))
    cf0 = ax0.contourf(x, y, u, np.arange(-0.2, 1.1, .05),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, pad=0.03, aspect=25, format='%.0e')
    plt.title("u", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Rectangle((0, 1), 1, 1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar0.ax.tick_params(labelsize=35)
    plt.show()

    ########################### Y-velocity

    fig0, ax0 = plt.subplots(1, 1,figsize=(12,10))
    cf0 = ax0.contourf(x, y, v, np.arange(-0.2, 1.1, .05),
                   extend='both',cmap='rainbow')
    cbar0 = plt.colorbar(cf0, pad=0.03, aspect=25, format='%.0e')
    plt.title("v", fontdict = font1)
    plt.xlabel("x", fontdict = font1)
    plt.ylabel("y", fontdict = font1)
    ax0.add_patch(Rectangle((0, 1), 1, 1,color="black"))
    plt.tick_params(axis='both', which='major', labelsize=35)
    cbar0.ax.tick_params(labelsize=35)
    plt.show()


