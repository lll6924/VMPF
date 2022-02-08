from model import StateSpaceModel
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import scipy.stats
class StochasticVolatilityParameters:
    def __init__(self,
                 d_x = 22,
                 scale = 0.5,
                 T = 119,
                 beta_form = 'triangular',
                 **kwargs):
        d_x = int(d_x)
        self.d_x = d_x
        self.T = T
        self.mu = tf.Variable(scale * tf.random.normal((d_x,), dtype=tf.float32), trainable=True,
                              name='mu', dtype=tf.float32)
        self.phi = tf.Variable(scale * tf.random.normal((d_x,), dtype=tf.float32), trainable=True,
                               name='phi', dtype=tf.float32)
        self.Q = tf.Variable(scale * tf.random.normal((d_x,), dtype=tf.float32), trainable=True,
                                 name='Q', dtype=tf.float32)
        self.beta_form = beta_form
        if beta_form == 'triangular':
            self.beta = tf.Variable(scale * tf.random.normal((d_x * (d_x + 1) // 2,), dtype=tf.float32), trainable=True,
                                    name='beta', dtype=tf.float32)
        else:
            self.beta = tf.Variable(scale * tf.random.normal((d_x,), dtype=tf.float32), trainable=True,
                                    name='beta', dtype=tf.float32)
        self.trainable_variables = [self.mu, self.phi, self.Q, self.beta]

    def get_trainable_variables(self):
        return self.trainable_variables

class StochasticVolatilityData:
    def __init__(self, **kwargs):
        self.parameters = None
    def get_data(self):
        y = []
        with open('dataset/FRB_H10.csv') as f:
            data = f.readlines()
            for d in data[6:]:
                d = d.split(',')[1:-1]
                d = [float(i) for i in d]
                y.append(d)
        y = np.asarray(y)
        return None,np.log(y[1:])-np.log(y[:-1])


class StochasticVolatility(StateSpaceModel):
    def __init__(self, **kwargs):
        self.parameters =  StochasticVolatilityParameters(**kwargs)

    def logf(self, time, x_t, given=None):
        mu = self.parameters.mu+tf.sigmoid(self.parameters.phi)*(given-self.parameters.mu)
        dist = tfd.MultivariateNormalDiag(mu, tf.exp(self.parameters.Q))

        ans = dist.log_prob(x_t)
        return ans

    def prior_parameters(self, time, given = None):
        mu = self.parameters.mu+tf.sigmoid(self.parameters.phi)*(given-self.parameters.mu)
        return mu, self.parameters.Q

    def logg(self,y_t, x_t):
        mu = tf.zeros(self.parameters.d_x)
        if self.parameters.beta_form == 'triangular':
            beta_tril = tfp.math.fill_triangular(self.parameters.beta)
            beta_tril = tf.linalg.set_diag(beta_tril, tf.exp(tf.linalg.diag_part(beta_tril)))
            tril = tf.multiply(beta_tril, tf.expand_dims(tf.exp(x_t/2),axis=-1))
            dist = tfd.MultivariateNormalTriL(mu, tril)
        else:
            dist = tfd.MultivariateNormalDiag(mu, tf.exp(self.parameters.beta)*tf.exp(x_t/2))
        ans = dist.log_prob(y_t)
        return ans

    def initial_state(self,N):
        return tf.expand_dims(tf.tile(tf.expand_dims(self.parameters.mu,0),(4,1)),0)

    def initial_state_vmpf(self):
        return self.parameters.mu

    def get_trainable_variables(self):
        return self.parameters.get_trainable_variables()

