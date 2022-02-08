import tensorflow as tf
import tensorflow_probability as tfp
from proposal import GaussianProposal
import numpy as np
from tensorflow_probability import distributions as tfd

class DNN4TMC(GaussianProposal):
    def __init__(self,
                 model,
                 **kwargs):

        self.y_net = tf.keras.Sequential()
        self.y_net.add(tf.keras.layers.Dense(model.parameters.d_l))
        self.y_net.add(tf.keras.layers.LeakyReLU())
        self.y_net.add(tf.keras.layers.Dense(model.parameters.d_x*2))
        self.y_net.build((None,model.parameters.d_y))
        self.trainable_variables = self.y_net.trainable_variables
        self.model = model


    def LocAndScale(self, t, given,y_t):
        raise NotImplementedError()

    def vmpf_proposal(self, t, given, y_t, W, reparameterize = True):
        y_t = tf.expand_dims(y_t,axis=0)
        y_loc_scale = self.y_net(y_t)

        y_mu = y_loc_scale[0, :self.model.parameters.d_x]
        y_logstd = y_loc_scale[0, self.model.parameters.d_x:]
        dist = tfd.MultivariateNormalDiag(y_mu,tf.exp(y_logstd))
        return dist

    def get_trainable_variables(self):
        return self.trainable_variables

    def diagnal(self):
        return True
