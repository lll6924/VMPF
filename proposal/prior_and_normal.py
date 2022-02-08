import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from proposal import GaussianProposal
import numpy as np
class PriorAndNormal(GaussianProposal):
    def __init__(self,
                 model,
                 scale = 0.5,
                 **kwargs):
        self.mu = tf.Variable(scale * tf.random.normal((model.parameters.T,model.parameters.d_x,), dtype=tf.float32), trainable=True,
                         name='mu', dtype=tf.float32)
        self.logsigma2 = tf.Variable(scale * tf.random.normal((model.parameters.T,model.parameters.d_x,), dtype=tf.float32), trainable=True,
                                name='logsigma2', dtype=tf.float32)
        self.trainable_variables = [self.mu, self.logsigma2]
        self.model = model

    def LocAndScale(self, t, given,y_t):
        mu1, Sigma1 = self.model.prior_parameters(t,given)
        mu2 = tf.squeeze(tf.slice(self.mu,[t,0],[1,-1]),axis=0)
        Sigma2 = tf.squeeze(tf.slice(self.logsigma2,[t,0],[1,-1]),axis=0)
        Sigma = -tf.reduce_logsumexp([-Sigma1,-Sigma2],axis=0)

        mu = mu1*tf.exp(Sigma-Sigma1) + mu2*tf.exp(Sigma-Sigma2)
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