import tensorflow as tf
#from .layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for the steady Navier-Stokes equation.
    Attributes:
        network: keras network model with input (x, y) and output (psi, p).
        rho: density.
        nu: viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, rho=1, mu=0.01):
        """
        Args:
            network: keras network model with input (x, y) and output (u, v, p).
            rho: density.
            nu: viscosity.
        """

        self.network = network
        self.rho = rho
        self.mu = mu
        self.grads = GradientLayer(self.network)

    def build(self):
        """
        Build a PINN model for the steady Navier-Stokes equation.
        Returns:
            PINN model for the steady Navier-Stokes equation with
                input: [ (x, y) relative to equation,
                         (x, y) relative to boundary condition ],
                output: [ (u, v) relative to equation (must be zero),
                          (u, v, p) relative to boundary conditions ]
        """

        # equation input: (x, y)
        xy_eqn = tf.keras.layers.Input(shape=(2,))
        # boundary condition
        ##xy_bnd = tf.keras.layers.Input(shape=(3,))
        xy_in = tf.keras.layers.Input(shape=(2,))
        xy_out = tf.keras.layers.Input(shape=(2,))
        xy_w1 = tf.keras.layers.Input(shape=(2,))
        xy_w2 = tf.keras.layers.Input(shape=(2,))
        xy_w3 = tf.keras.layers.Input(shape=(2,))
        xy_w4 = tf.keras.layers.Input(shape=(2,))
        xy_box = tf.keras.layers.Input(shape=(2,))

        # compute gradients relative to equation
        p_grads, u_grads, v_grads = self.grads(xy_eqn)
        _, p_x, p_y = p_grads
        u, u_x, u_y, u_xx, u_yy = u_grads
        v, v_x, v_y, v_xx, v_yy = v_grads
        # compute equation loss
        u_eqn =  u*u_x + v*u_y + p_x/self.rho - self.mu*(u_xx + u_yy) / self.rho
        v_eqn =  u*v_x + v*v_y + p_y/self.rho - self.mu*(v_xx + v_yy) / self.rho
        uv_eqn = u_x + v_y
        uv_eqn = tf.concat([u_eqn, v_eqn, uv_eqn], axis=-1)

        # compute gradients relative to boundary condition
        p_r, u_grads_r, v_grads_r = self.grads(xy_out)
        uv_out = tf.concat([p_r[0], p_r[0], p_r[0]], axis=-1)

        p_l, u_grads_l, v_grads_l = self.grads(xy_w1)
        uv_w1 = tf.concat([u_grads_l[0], v_grads_l[0], p_l[2]], axis=-1)
        
        p_l, u_grads_l, v_grads_l = self.grads(xy_w2)
        uv_w2 = tf.concat([u_grads_l[0], v_grads_l[0], p_l[1]], axis=-1)
        
        p_l, u_grads_l, v_grads_l = self.grads(xy_w3)
        uv_w3 = tf.concat([u_grads_l[0], v_grads_l[0], p_l[1]], axis=-1)
        
        p_l, u_grads_l, v_grads_l = self.grads(xy_w4)
        uv_w4 = tf.concat([u_grads_l[0], v_grads_l[0], p_l[2]], axis=-1)
        
        p_inn, u_inn, v_inn = self.grads(xy_in)
        uv_in = tf.concat([u_inn[0], v_inn[0], u_inn[0]], axis=-1)

        # build the PINN model for the steady Navier-Stokes equation
        return tf.keras.models.Model(
            inputs=[xy_eqn, xy_w1, xy_w2, xy_w3, xy_w4, xy_out, xy_in, xy_box], outputs=[uv_eqn, uv_in, uv_out, uv_w1, uv_w2, uv_w3, uv_w4])
