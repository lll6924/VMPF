import attr
import tensorflow as tf
import tensorflow_probability as tfp

from filterflow.base import State
from filterflow.proposal import AdaptedProposalModel
from filterflow.observation import AdaptedObservationModel
from filterflow.transition import AdaptedTransitionModel
from filterflow.smc import SMC
from filterflow.smc_multiple_data import  SMCMultipleData

tfd = tfp.distributions


def make_filter(model, proposal, data, N, resampling_method,
                resampling_criterion):

    observation_model = AdaptedObservationModel(model)

    transition_model = AdaptedTransitionModel(model)
    proposal_model = AdaptedProposalModel(proposal, N)

    return SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)

def make_filter_multiple_data(model, proposal, data, N, resampling_method,
                resampling_criterion):

    observation_model = AdaptedObservationModel(model)

    transition_model = AdaptedTransitionModel(model)
    proposal_model = AdaptedProposalModel(proposal, N)

    return SMCMultipleData(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)
