from model import StateSpaceModel
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import numpy as np
import scipy.stats
class LinearGaussianParameters:
    # The model is
    # x_t = \mathcal{N}(Ax_{t-1}, diag(Q))
    # y_t = \mathcal{N}(Cx_t, diag(R))
    def __init__(self,
                 d_x = 10,
                 d_y = 3,
                 T = 10,
                 mod = 'tensor',
                 alpha = .42,
                 r = 1.,
                 obs = 'dense',
                 **kwargs):
        d_x = int(d_x)
        d_y = int(d_y)
        self.d_x = d_x
        self.d_y = d_y
        A = np.zeros((d_x, d_x))
        for i in range(d_x):
            for j in range(d_x):
                A[i, j] = alpha ** (abs(i - j) + 1)
        np.random.seed(0)
        Q = np.eye(d_x)
        C = np.zeros((d_y, d_x))
        if obs == 'sparse':
            C[:d_y, :d_y] = np.eye(d_y)
        else:
            C = np.random.normal(size=(d_y, d_x))
        R = r * np.eye(d_y)
        self.A = A
        self.Q = Q
        self.C = C
        self.R = R
        self.T = T

        if mod == 'tensor':
            self.A = tf.convert_to_tensor(self.A,dtype=tf.float32)
            self.Q = tf.convert_to_tensor(self.Q,dtype=tf.float32)
            self.C = tf.convert_to_tensor(self.C,dtype=tf.float32)
            self.R = tf.convert_to_tensor(self.R,dtype=tf.float32)
    def get_trainable_variables(self):
        return []

class LinearGaussianData:
    def __init__(self, **kwargs):
        self.parameters = LinearGaussianParameters(mod = 'numpy',**kwargs)
    def get_data(self):
        d_x = self.parameters.d_x
        d_y = self.parameters.d_y
        T = self.parameters.T

        x_true = np.zeros((T, d_x))
        y_true = np.zeros((T, d_y))
        np.random.seed(0)
        for t in range(T):
            if t > 0:
                x_true[t, :] = np.random.multivariate_normal(np.dot(self.parameters.A, x_true[t - 1, :]), self.parameters.Q)
            else:
                x_true[0, :] = np.random.multivariate_normal(np.zeros(d_x), self.parameters.Q)
            y_true[t, :] = np.random.multivariate_normal(np.dot(self.parameters.C, x_true[t, :]), self.parameters.R)
        return x_true, y_true

class KalmanFilter:
    def __init__(self, parameters, data):
        self.data = data
        self.parameters = parameters

    def log_p_y(self):
        d_x = self.parameters.d_x
        d_y = self.parameters.d_y
        T = self.parameters.T
        y_true = self.data
        mu = np.zeros((d_x))
        Sigma = np.zeros((d_x,d_x))
        ans = 0
        for t in range(T):
            mu = np.matmul(self.parameters.A,mu)
            Sigma = np.matmul(np.matmul(self.parameters.A,Sigma),self.parameters.A.transpose())+self.parameters.Q
            Sigma_yy = np.matmul(np.matmul(self.parameters.C,Sigma),self.parameters.C.transpose())+self.parameters.R
            mu_y = np.matmul(self.parameters.C,mu)
            ans += scipy.stats.multivariate_normal(mean=mu_y,cov=Sigma_yy).logpdf(y_true[t])
            K = np.matmul(np.matmul(Sigma,self.parameters.C.transpose()),np.linalg.inv(Sigma_yy))
            mu = mu + np.matmul(K,y_true[t]-np.matmul(self.parameters.C,mu))
            Sigma = np.matmul(np.eye(d_x)-np.matmul(K,self.parameters.C),Sigma)
        return ans

class LinearGaussian(StateSpaceModel):
    def __init__(self, **kwargs):
        self.parameters = LinearGaussianParameters(**kwargs)

    def logf(self, time, x_t, given=None):
        mu = tf.tensordot(self.parameters.A, given,axes = [[-1],[-1]])
        mu = tf.transpose(mu, tuple(range(1,len(mu.shape)))+(0,))
        dist = tfd.MultivariateNormalFullCovariance(mu, self.parameters.Q)
        ans = dist.log_prob(x_t)
        return ans

    def logg(self,y_t, x_t):
        mu = tf.tensordot(self.parameters.C,  x_t, axes= [[-1],[-1]])
        mu = tf.transpose(mu, tuple(range(1,len(mu.shape)))+(0,))
        dist = tfd.MultivariateNormalFullCovariance(mu, self.parameters.R)
        ans = dist.log_prob(y_t)
        return ans

    def initial_state(self,N):
        return tf.zeros((1,N,self.parameters.d_x))

    def initial_state_vmpf(self):
        return tf.zeros(self.parameters.d_x,dtype=tf.float32)

    def get_trainable_variables(self):
        return self.parameters.get_trainable_variables()