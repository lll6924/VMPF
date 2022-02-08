import tensorflow as tf
import tensorflow_probability as tfp
from proposal import GaussianProposal
import numpy as np
from tensorflow_probability import distributions as tfd

class DenseNeuralNetwork(GaussianProposal):
    def __init__(self,
                 model,
                 **kwargs):
        self.x_net = tf.keras.Sequential()
        self.x_net.add(tf.keras.layers.Dense(model.parameters.d_l))
        self.x_net.add(tf.keras.layers.LeakyReLU())
        self.x_net.add(tf.keras.layers.Dense(model.parameters.d_x*2))
        self.x_net.build((None,model.parameters.d_x))

        self.y_net = tf.keras.Sequential()
        self.y_net.add(tf.keras.layers.Dense(model.parameters.d_l))
        self.y_net.add(tf.keras.layers.LeakyReLU())
        self.y_net.add(tf.keras.layers.Dense(model.parameters.d_x*2))
        self.y_net.build((None,model.parameters.d_y))
        #self.q_net.summary()
        self.trainable_variables = self.x_net.trainable_variables
        self.trainable_variables.extend(self.y_net.trainable_variables)
        self.model = model


    def LocAndScale(self, t, given,y_t):
        shapediff  = given.shape[:len(given.shape) - len(y_t.shape)]
        y_t = tf.broadcast_to(y_t,shapediff + y_t.shape)

        x_loc_scale = self.x_net(given)
        y_loc_scale = self.y_net(y_t)

        x_mu = x_loc_scale[..., :self.model.parameters.d_x]
        x_logstd = x_loc_scale[..., self.model.parameters.d_x:]
        y_mu = y_loc_scale[..., :self.model.parameters.d_x]
        y_logstd = y_loc_scale[..., self.model.parameters.d_x:]

        logstd = -tf.reduce_logsumexp([-x_logstd,-y_logstd],axis=0)


        mu = x_mu*tf.exp(logstd-x_logstd) + y_mu*tf.exp(logstd-y_logstd)
        logstd = tf.broadcast_to(logstd,mu.shape)
        return mu, logstd

    def vmpf_proposal(self, t, given, y_t, W, reparameterize = True):
        mu, Sigma = self.LocAndScale(t, given, y_t)
        dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=W),
            components_distribution=tfd.Independent(tfd.Normal(
                loc=mu,
                scale=tf.exp(Sigma))),
            reparameterize=reparameterize)
        return dist

    def get_trainable_variables(self):
        return self.trainable_variables

    def diagnal(self):
        return True
