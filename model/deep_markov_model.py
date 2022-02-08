from model import StateSpaceModel
from tensorflow_probability import distributions as tfd
import tensorflow as tf
import numpy as np
import scipy.stats
import pickle

class DeepMarkovModelParameters:
    def __init__(self,
                 d_x = 88,
                 d_y = 88,
                 d_l = 32):
        self.d_x = d_x
        self.d_y = d_y
        self.d_l = d_l
        self.f_net = tf.keras.Sequential()
        self.f_net.add(tf.keras.layers.Dense(d_l))
        self.f_net.add(tf.keras.layers.LeakyReLU())
        self.f_net.add(tf.keras.layers.Dense(2*d_x))
        self.g_net = tf.keras.Sequential()
        self.g_net.add(tf.keras.layers.Dense(d_l))
        self.g_net.add(tf.keras.layers.LeakyReLU())
        self.g_net.add(tf.keras.layers.Dense(d_y))
        self.f_net.build((None,d_x))
        self.g_net.build((None,d_x))
        #self.f_net.summary()
        #self.g_net.summary()
        self.trainable_variables = []
        self.trainable_variables.extend(self.f_net.trainable_variables)
        self.trainable_variables.extend(self.g_net.trainable_variables)
    def get_trainable_variables(self):
        return self.trainable_variables


dataset_mapping = {'Nottingham':'Nottingham.pickle',
                   'JSB':'JSB Chorales.pickle',
                   'MuseData':'MuseData.pickle',
                   'Piano-midi.de':'Piano-midi.de.pickle'}

def get_data(dataset, min_note=21, max_note=108):
    ret = []
    for sample in dataset:
        data = np.zeros((len(sample),max_note-min_note+1))
        for i in range(len(sample)):
            for s in sample[i]:
                data[i][s-min_note]=1
        ret.append(data)
    return ret

class DeepMarkovModelData:
    def __init__(self, parameters = None, dataset='JSB'):
        assert(dataset in ['Nottingham','JSB','MuseData','Piano-midi.de'])
        self.dataset=dataset
        self.parameters = parameters
    def get_data(self):

        with open('dataset/'+dataset_mapping[self.dataset], 'rb') as f:
            obj = pickle.load(f)
            test = obj['test']
            train = obj['train']
            valid = obj['valid']
            test_data = get_data(test)
            train_data = get_data(train)
            valid_data = get_data(valid)
        return train_data,valid_data,test_data


class DeepMarkovModel(StateSpaceModel):
    def __init__(self, parameters = DeepMarkovModelParameters()):
        self.parameters = parameters

    def prior_parameters(self,x_t = None):
        loc_scale = self.parameters.f_net(x_t)

        mu = loc_scale[...,:self.parameters.d_x]
        logstd = loc_scale[...,self.parameters.d_x:]
        return mu, logstd

    def logf(self, time, x_t, given = None):
        mu, logstd = self.prior_parameters(given)
        dist = tfd.MultivariateNormalDiag(mu, tf.exp(logstd))
        ans = dist.log_prob(x_t)
        return ans

    def logg(self,y_t, x_t):
        logs = self.parameters.g_net(x_t)
        dist = tfd.Bernoulli(logits = logs)
        ans = tf.reduce_sum(dist.log_prob(y_t),axis=-1)
        return ans

    def initial_state(self,N):
        return tf.zeros((1,N,self.parameters.d_x))

    def initial_state_vmpf(self):
        return tf.zeros((1,self.parameters.d_x),dtype=tf.float32)

    def get_trainable_variables(self):
        return self.parameters.get_trainable_variables()