from algorithm import Algorithm
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from model import DeepMarkovModel, DeepMarkovModelData, DeepMarkovModelParameters
from proposal import DenseNeuralNetwork
import numpy as np

class VariationalSequentialMonteCarloMultipleData(Algorithm):
    def __init__(self, d_h=32, proposal=DenseNeuralNetwork, N=4, data = DeepMarkovModelData(), reparameterize=True, mode = 'any'):
        self.ssm = DeepMarkovModel(DeepMarkovModelParameters(d_l=d_h))
        self.proposal = proposal(model = self.ssm)
        self.trainable_variables = self.proposal.get_trainable_variables()
        self.trainable_variables.extend(self.ssm.get_trainable_variables())
        self.N = N
        self.train, self.valid, self.test = data.get_data()
        self.W = self.X = None
        self.mode = mode
        self.reparameterize = reparameterize

    def get_trainable_variables(self):
        return self.trainable_variables

    @tf.function
    def init(self,y0, N = None):
        if N is None:
            N = self.N
        mean, sigma = self.proposal.LocAndScale(0,self.ssm.initial_state_vmpf(), y0)
        mean = mean[0]
        sigma = sigma[0]
        dist = tfd.MultivariateNormalDiag(mean, tf.exp(sigma))
        x0 = dist.sample((N),reparameterization_type=tfd.FULLY_REPARAMETERIZED if self.reparameterize else tfd.NOT_REPARAMETERIZED)
        logw0 = self.ssm.logf(0, x0, self.ssm.initial_state_vmpf()) + self.ssm.logg(y0, x0) - dist.log_prob(x0)

        ave_w = tfp.math.reduce_logmeanexp(logw0)
        normalizer = tf.reduce_logsumexp(logw0)
        logw0_bar = logw0 - normalizer
        return x0, logw0, logw0_bar, ave_w

    @tf.function
    def step(self, x, w, w_bar, y_t, N = None):
        if N is None:
            N = self.N
        a = tfp.experimental.mcmc.resample_stratified(w_bar, event_size=1, sample_shape=(N,))[0]
        x_hat = tf.gather(x, a)
        means, sigma = self.proposal.LocAndScale(None, x_hat, y_t)
        dist = tfd.MultivariateNormalDiag(means, tf.exp(sigma))
        xt = dist.sample(
            reparameterization_type=tfd.FULLY_REPARAMETERIZED if self.reparameterize else tfd.NOT_REPARAMETERIZED)
        logwt = self.ssm.logf(None, xt, x_hat) + self.ssm.logg(y_t, xt) - dist.log_prob(xt)
        ave_w = tfp.math.reduce_logmeanexp(logwt)
        normalizer = tf.reduce_logsumexp(logwt)
        logwt_bar = logwt - normalizer
        return a, xt, logwt, logwt_bar, ave_w

    def log_p_hat(self, y):
        N = self.N
        x, logw, logw_bar, ave_w = self.init(y[0],N)
        ans = ave_w
        for t in range(1, len(y)):
            a, x, logw, logw_bar, ave_w = self.step(x, logw, logw_bar,y[t],N)
            ans += ave_w
        return ans

    def loss(self, y = None):
        return -self.log_p_hat(y)

    @staticmethod
    def getW(self):
        return self.W

    @staticmethod
    def getX(self):
        return self.X

    @staticmethod
    def getA(self):
        return self.A
