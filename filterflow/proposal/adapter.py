import abc
import attr

import tensorflow as tf

from filterflow.base import State, Module
from filterflow.proposal.base import ProposalModelBase
from tensorflow_probability import distributions as tfd


class AdaptedProposalModel(ProposalModelBase):
    def __init__(self, proposal, N):
        super(AdaptedProposalModel, self).__init__()
        self.proposal = proposal
        self.N = N
    def propose(self, state: State, inputs: tf.Tensor, observation: tf.Tensor, seed=None):
        """Interface method for particle proposal

        :param state: State
            previous particle filter state
        :param inputs: tf.Tensor
            Control variables (time elapsed, some environment variables, etc)
        :param observation: tf.Tensor
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: State
        """
        mu, Sigma = self.proposal.LocAndScale(state.t, tf.squeeze(state.particles,axis=0), observation)
        dist = tfd.MultivariateNormalDiag(mu, tf.exp(Sigma))
        particles = dist.sample((1,))
        return attr.evolve(state, particles=particles)

    def loglikelihood(self, proposed_state: State, state: State, inputs: tf.Tensor, observation: tf.Tensor):
        """Interface method for particle proposal
        :param proposed_state: State
            proposed state
        :param state: State
            previous particle filter state
        :param inputs: tf.Tensor
            Control variables (time elapsed, some environment variables, etc)
        :param observation: tf.Tensor
            Look ahead observation for adapted particle proposal
        :return: proposed State
        :rtype: tf.Tensor
        """
        mu, Sigma = self.proposal.LocAndScale(state.t, tf.squeeze(state.particles,axis=0), observation)
        dist = tfd.MultivariateNormalDiag(mu, tf.exp(Sigma))

        return dist.log_prob(tf.squeeze(proposed_state.particles,axis=0))
