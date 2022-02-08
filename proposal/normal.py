import tensorflow as tf
from proposal import GaussianProposal
import numpy as np
from tensorflow_probability import distributions as tfd

class NormalProposal(GaussianProposal):
    def __init__(self,
                 model,
                 scale = 0.5,
                 **kwargs):
        self.mu = tf.Variable(scale * tf.random.normal((model.parameters.T, model.parameters.d_x,), dtype=tf.float32),
                              trainable=True, name='mu', dtype=tf.float32)
        self.beta = tf.Variable(1. + scale * tf.random.normal((model.parameters.T, model.parameters.d_x,), dtype=tf.float32),
                                trainable=True, name='beta', dtype=tf.float32)
        self.logsigma2 = tf.Variable(scale * tf.random.normal((model.parameters.T, model.parameters.d_x,), dtype=tf.float32),
                                     trainable=True, name='logsigma2', dtype=tf.float32)
        self.trainable_variables = [self.mu, self.beta, self.logsigma2]
        self.model = model

    def LocAndScale(self, t, given,y_t):
        mum = tf.tensordot(self.model.parameters.A,given, axes =[[-1],[-1]])
        mum = tf.transpose(mum, tuple(range(1, len(mum.shape))) + (0,))
        mut = tf.squeeze(tf.slice(self.mu,[t,0],[1,-1]),axis=0)
        betat = tf.squeeze(tf.slice(self.beta,[t,0],[1,-1]),axis=0)
        logsigma2t = tf.squeeze(tf.slice(self.logsigma2,[t,0],[1,-1]),axis=0)
        mu = mum*betat+mut
        Sigma = logsigma2t
        Sigma = tf.broadcast_to(Sigma,mu.shape)
        return mu, Sigma

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