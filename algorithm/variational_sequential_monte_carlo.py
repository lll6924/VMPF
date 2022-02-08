from algorithm import Algorithm
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.python.ops import math_ops
from model import LinearGaussian, LinearGaussianData, KalmanFilter
from proposal import DenseNeuralNetwork, NormalProposal

class VariationalSequentialMonteCarlo(Algorithm):
    def __init__(self, ssm=LinearGaussian, proposal=NormalProposal, N=4, data = LinearGaussianData, reparameterize = True, adaptive_resample=False, **kwargs):
        self.ssm = ssm(**kwargs)
        self.proposal = proposal(model = self.ssm)
        self.trainable_variables = self.proposal.get_trainable_variables()
        self.trainable_variables.extend(self.ssm.get_trainable_variables())
        self.N = N
        self.data = data(**kwargs)
        self.true_x, self.true_y = self.data.get_data()
        if isinstance(self.ssm,LinearGaussian):
            kf = KalmanFilter(self.data.parameters,self.true_y)
            print('logpy (KF): ',kf.log_p_y())
        if not type(self.true_y) == tuple:
            self.y = tf.convert_to_tensor(self.true_y, dtype=tf.float32)
        else:
            self.y = None
            self.train, self.valid, self.test = self.true_y
        self.W = self.X = None
        self.reparameterize = reparameterize
        self.adaptive_resample = adaptive_resample


    def get_trainable_variables(self):
        return self.trainable_variables

    @tf.function
    def log_p_hat(self, y = None, N = None):
        if y is None:
            y = self.y
        if N is None:
            N = self.N
        mean, sigma = self.proposal.LocAndScale(0,self.ssm.initial_state_vmpf(), y[0])
        dist = tfd.MultivariateNormalDiag(mean, tf.exp(sigma))
        x = dist.sample((N),reparameterization_type=tfd.FULLY_REPARAMETERIZED if self.reparameterize else tfd.NOT_REPARAMETERIZED)
        logw = self.ssm.logf(0, x, self.ssm.initial_state_vmpf()) + self.ssm.logg(y[0], x) - dist.log_prob(x)

        ans = 0.#tfp.math.reduce_logmeanexp(logw)
        normalizer = tf.reduce_logsumexp(logw)
        logw_bar = logw - normalizer
        for t in range(1, len(y)):
            ESS = 1. / tf.reduce_sum(tf.exp(logw_bar*2)) / N
            if self.adaptive_resample==False or ESS < 0.5:
                ans += tfp.math.reduce_logmeanexp(logw)
                a = tfp.experimental.mcmc.resample_stratified(logw_bar, event_size=1, sample_shape=(N,))[0]
                x_hat = tf.gather(x,a)
                means, sigma = self.proposal.LocAndScale(t, x_hat, y[t])
                dist = tfd.MultivariateNormalDiag(means, tf.exp(sigma))
                x = dist.sample(
                    reparameterization_type=tfd.FULLY_REPARAMETERIZED if self.reparameterize else tfd.NOT_REPARAMETERIZED)
                logw = self.ssm.logf(t, x, x_hat) + self.ssm.logg(y[t], x) - dist.log_prob(x)
            else:
                x_hat = x
                means, sigma = self.proposal.LocAndScale(t, x_hat, y[t])
                dist = tfd.MultivariateNormalDiag(means, tf.exp(sigma))
                x = dist.sample(
                    reparameterization_type=tfd.FULLY_REPARAMETERIZED if self.reparameterize else tfd.NOT_REPARAMETERIZED)
                logw += self.ssm.logf(t, x, x_hat) + self.ssm.logg(y[t], x) - dist.log_prob(x)



            normalizer = tf.reduce_logsumexp(logw)
            logw_bar = logw - normalizer
        ans += tfp.math.reduce_logmeanexp(logw)
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
