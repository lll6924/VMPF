import tensorflow as tf
from proposal import GaussianProposal
import numpy as np
from tensorflow_probability import distributions as tfd

class Normal4TMC(GaussianProposal):
    def __init__(self,
                 model,
                 scale = 0.5,
                 code = 'vmpf',
                 **kwargs):
        self.mu = tf.Variable(scale * tf.random.normal((model.parameters.T, model.parameters.d_x,), dtype=tf.float32),
                              trainable=True, name='mu', dtype=tf.float32)
        self.logsigma2 = tf.Variable(scale * tf.random.normal((model.parameters.T, model.parameters.d_x,), dtype=tf.float32),
                                     trainable=True, name='logsigma2', dtype=tf.float32)
        self.trainable_variables = [self.mu, self.logsigma2]
        self.model = model

    def LocAndScale(self, t, given,y_t):
        raise NotImplementedError()

    def vmpf_proposal(self, t, given, y_t, W, reparameterize = True):
        mut = tf.squeeze(tf.slice(self.mu,[t,0],[1,-1]),axis=0)
        logsigma2t = tf.squeeze(tf.slice(self.logsigma2,[t,0],[1,-1]),axis=0)
        dist = tfd.MultivariateNormalDiag(mut,tf.exp(logsigma2t))
        return dist

    def get_trainable_variables(self):
        return self.trainable_variables

    def diagnal(self):
        return True