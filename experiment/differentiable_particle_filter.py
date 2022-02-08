# adapted from https://github.com/JTT94/filterflow/blob/master/scripts/stochastic_volatility.py

import enum
import os
import sys
import time
from datetime import datetime

sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import flags, app
import tensorflow_probability as tfp
from tensorflow_probability.python.internal.samplers import split_seed
from tqdm import tqdm

# tf.function = lambda z: z

from filterflow.base import State
from filterflow.resampling import MultinomialResampler, SystematicResampler, StratifiedResampler, RegularisedTransform, \
    CorrectedRegularizedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from filterflow.resampling.differentiable import PartiallyCorrectedRegularizedTransform
from filterflow.resampling.differentiable.loss import SinkhornLoss
from filterflow.resampling.differentiable.optimized import OptimizedPointCloud
from filterflow.resampling.differentiable.optimizer.sgd import SGD
from filterflow.models.adapter import make_filter
import quandl

from algorithm import VariationalMarginalParticleFilter, VariationalSequentialMonteCarlo, VariationalMarginalParticleFilterBiasedGradients, ImportanceWeightedVariationalInference
from proposal import DenseNeuralNetwork, NormalProposal, PriorAndNormal
from model import StochasticVolatility, LinearGaussian, StochasticVolatilityData, LinearGaussianData, StochasticVolatilityParameters, KalmanFilter
from utils import Logger

class ResamplingMethodsEnum(enum.IntEnum):
    MULTINOMIAL = 0
    SYSTEMATIC = 1
    STRATIFIED = 2
    REGULARIZED = 3
    VARIANCE_CORRECTED = 4
    OPTIMIZED = 5
    CORRECTED = 6


def resampling_method_factory(resampling_method_enum, resampling_kwargs):
    if resampling_method_enum == ResamplingMethodsEnum.MULTINOMIAL:
        resampling_method = MultinomialResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.SYSTEMATIC:
        resampling_method = SystematicResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.STRATIFIED:
        resampling_method = StratifiedResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.REGULARIZED:
        resampling_method = RegularisedTransform(**resampling_kwargs)
    elif resampling_method_enum == ResamplingMethodsEnum.VARIANCE_CORRECTED:
        regularized_resampler = RegularisedTransform(**resampling_kwargs)
        resampling_method = PartiallyCorrectedRegularizedTransform(regularized_resampler)
    elif resampling_method_enum == ResamplingMethodsEnum.OPTIMIZED:

        lr = resampling_kwargs.pop('lr', resampling_kwargs.pop('learning_rate', 0.1))
        decay = resampling_kwargs.pop('lr', 0.95)
        symmetric = resampling_kwargs.pop('symmetric', True)
        loss = SinkhornLoss(**resampling_kwargs, symmetric=symmetric)

        optimizer = SGD(loss, lr=lr, decay=decay)

        regularized_resampler = RegularisedTransform(**resampling_kwargs)
        intermediate_resampling_method = PartiallyCorrectedRegularizedTransform(regularized_resampler)

        resampling_method = OptimizedPointCloud(optimizer, intermediate_resampler=intermediate_resampling_method)
    elif resampling_method_enum == ResamplingMethodsEnum.CORRECTED:
        resampling_method = CorrectedRegularizedTransform(**resampling_kwargs)
    else:
        raise ValueError(f'resampling_method_name {resampling_method_enum} is not a valid ResamplingMethodsEnum')
    return resampling_method


@tf.function
def routine(pf, initial_state, observations_dataset, T, variables, seed):
    with tf.GradientTape() as tape:
        tape.watch(variables)
        final_state = pf(initial_state, observations_dataset, T, seed=seed, return_final=True)
        res = -tf.reduce_mean(final_state.log_likelihoods)
    return res, tape.gradient(res, variables), tf.reduce_mean(final_state.ess)


def get_gradient_descent_function():
    # This is a trick because tensorflow doesn't allow you to create variables inside a decorated function

    def gradient_descent(pf, initial_state, observations_dataset, T, n_iter, optimizers, variables,
                         initial_values, change_seed, seed, large_initial_state, surrogate_smc, logdir, track_gradients, evaluate_period = 100, evaluation_n = 100, final_evaluation_n = 1000):
        reset_operations = [k.assign(v) for k, v in zip(variables, initial_values)]
        loss = tf.TensorArray(dtype=tf.float32, size=1 + n_iter*len(optimizers), dynamic_size=False)
        ess = tf.TensorArray(dtype=tf.float32, size=1 + n_iter*len(optimizers), dynamic_size=False)

        filter_seed, seed = split_seed(seed, n=2, salt='gradient_descent')
        variable_number = np.sum([np.prod(v.shape) for v in variables])
        print('number of parameters: ', variable_number)
        with tf.control_dependencies(reset_operations):
            step = 1
            last_time = time.time()
            loss_data = []
            grad_var_data = []
            for optimizer in optimizers:
                for i in tf.range(n_iter):
                    # for var in variables:
                    #     tf.print(var)
                    loss_value, grads, average_ess = routine(pf, initial_state, observations_dataset, T, variables,
                                                             seed)
                    # tf.print(loss_value)

                    if change_seed:
                        filter_seed, seed = split_seed(filter_seed, n=2)

                    ess = ess.write(tf.cast(i, tf.int32), average_ess)
                    # for grad in grads:
                    # tf.print(tf.reduce_max(tf.abs(grad)))
                    #print(grads)
                    max_grad = tf.reduce_max([tf.reduce_max(tf.abs(grad)) for grad in grads])
                    # for grad in grads:
                    # tf.print(tf.reduce_max(tf.abs(grad)))
                    optimizer.apply_gradients(zip(grads, variables))
                    # tf.print('')
                    if (i+step)%evaluate_period == 0:
                        elbos = []
                        for _ in range(evaluation_n):
                            elbo, _, _ = routine(surrogate_smc, large_initial_state, observations_dataset, T, variables,
                                            seed)
                            elbos.append(elbo)
                        loss = loss.write(tf.cast(i + step, tf.int32), tf.reduce_mean(elbos))
                        current_time = time.time()
                        tf.print('Step', i+step, ', lr: ', optimizer.learning_rate, ', grads: ', max_grad, ', loss: ', tf.reduce_mean(elbos), ' time period: ', current_time - last_time)
                        loss_data.append({'iteration': i+step, 'lb': tf.reduce_mean(elbos).numpy()})
                        if track_gradients:
                            gradients_sum = np.zeros(variable_number)
                            gradients_squared_sum = np.zeros(variable_number)
                            for _ in range(evaluation_n):
                                _, grads, _ = routine(pf, initial_state, observations_dataset, T, variables, seed)
                                flattened_grads = np.concatenate([g.numpy().flatten() for g in grads])
                                gradients_sum += flattened_grads
                                gradients_squared_sum += np.square(flattened_grads)
                            gradients_mean = gradients_sum / evaluation_n
                            gradients_squared_mean = gradients_squared_sum / evaluation_n
                            gradients_var = gradients_squared_mean - np.square(gradients_mean)
                            mean_var = np.mean(gradients_var)
                            print('log mean var of gradients: ', np.log(mean_var))
                            grad_var_data.append({'iteration': i + step, 'mean_var': mean_var})
                        last_time = current_time
                step += n_iter
            np.savez_compressed(os.path.join(logdir, 'loss.npz'), loss=loss_data, grad_var_data=grad_var_data)
            elbos = []
            for _ in range(final_evaluation_n):
                elbo, _, _ = routine(surrogate_smc, large_initial_state, observations_dataset, T, variables,
                                     seed)
                elbos.append(elbo)
            print('final evaluation:', -tf.reduce_mean(elbos).numpy())

        return variables, loss.stack(), ess.stack()

    return gradient_descent


def compile_learning_rates(pf, initial_state, observations_dataset, T, variables, initial_values, n_iter,
                           optimizer_maker, learning_rates, filter_seed, change_seed, large_initial_state,
                           surrogate_smc, logdir, track_gradients):
    loss_profiles = []
    ess_profiles = []
    optimizers = []
    for learning_rate in tqdm(learning_rates):
        optimizer = optimizer_maker(learning_rate=learning_rate)
        optimizers.append(optimizer)

    gradient_descent_function = get_gradient_descent_function()
    final_variables, loss_profile, ess_profile = gradient_descent_function(pf, initial_state, observations_dataset,
                                                                               T, n_iter, optimizers, variables,
                                                                               initial_values, change_seed, filter_seed,
                                                                               large_initial_state, surrogate_smc, logdir, track_gradients)
    loss_profiles.append(-loss_profile.numpy() / T)
    ess_profiles.append(ess_profile.numpy())
    return loss_profiles, ess_profiles


def plot_losses(loss_profiles_df, filename, savefig, dx, dy, dense, T, change_seed):
    fig, ax = plt.subplots(figsize=(5, 5))
    loss_profiles_df.style.float_format = '${:,.1f}'.format
    loss_profiles_df.plot(ax=ax, legend=False)

    # ax.set_ylim(250, 700)
    ax.legend()
    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join('./charts/',
                                 f'stochvol_lr_loss_{filename}_dx_{dx}_dy_{dy}_dense_{dense}_T_{T}_change_seed_{change_seed}.png'))
    else:
        fig.suptitle(f'stochvol_loss_{filename}_dx_{dx}_dy_{dy}_dense_{dense}_T_{T}')
        plt.show()


def plot_losses_vs_ess(loss_profiles_df, ess_profiles_df, filename, savefig, M, n_particles,
                       change_seed, batch_size, n_iter, epsilon):
    fig, ax = plt.subplots(figsize=(5, 3))
    loss_profiles_df.style.float_format = '${:,.1f}'.format
    loss_profiles_df.plot(ax=ax, legend=False)

    ax.set_xlim(0, n_iter)

    ax1 = ax.twinx()
    ess_profiles_df.plot.area(ax=ax1, legend=False, linestyle='--', alpha=0.33, stacked=False)

    ax.set_ylim(8, 21)
    ax1.set_ylim(1, n_particles)

    # ax.legend()
    fig.tight_layout()
    filename = f'stochvol_diff_lr_loss_ess_{filename}_epsilon_{epsilon}_N_{n_particles}__M_{M}_change_seed_{change_seed}_batch_size_{batch_size}'
    if savefig:
        fig.savefig(os.path.join('./charts/',
                                 filename + '.png'))
        loss_profiles_df.to_csv(os.path.join('./tables/',
                                             filename + '.csv'))

    else:
        fig.suptitle(f'stochvol_loss_ess_{filename}_nfactors_M_{M}')
        plt.show()


def plot_variables(variables_df, filename, savefig):
    fig, ax = plt.subplots(figsize=(5, 5))
    variables_df.plot(ax=ax)
    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join('./charts/', f'stochvol_different_lr_variables_{filename}.png'))
    else:
        fig.suptitle(f'stochvol_different_lr_variables_{filename}')
        plt.show()


def main(resampling_method_value, resampling_neff, proposal, model, dataset, parameter_args, logdir, track_gradients,
         learning_rates=(1e-2,), resampling_kwargs=None, batch_size=1, n_particles=4,
         n_iter=50, savefig=False, filter_seed=0, use_xla=False, change_seed=True, ):
    print(parameter_args)
    model = model(**parameter_args)
    proposal = proposal(model=model, **parameter_args)
    data = dataset(**parameter_args)
    trainable_variables = proposal.get_trainable_variables()
    trainable_variables.extend(model.get_trainable_variables())
    true_x, true_y = data.get_data()
    if isinstance(model, LinearGaussian):
        kf = KalmanFilter(data.parameters, true_y)
        print('logpy (KF): ', kf.log_p_y())
    true_y = np.asarray(true_y,dtype=np.float32)
    #print(true_y)
    T = len(true_y)

    resampling_method_enum = ResamplingMethodsEnum(resampling_method_value)
    observation_dataset = tf.data.Dataset.from_tensor_slices(true_y)

    if resampling_kwargs is None:
        resampling_kwargs = {}

    if resampling_neff == 0.:
        resampling_criterion = NeverResample()
    elif resampling_neff == 1.:
        resampling_criterion = AlwaysResample()
    else:
        resampling_criterion = NeffCriterion(resampling_neff, True)

    if resampling_method_enum == ResamplingMethodsEnum.MULTINOMIAL:
        resampling_method = MultinomialResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.SYSTEMATIC:
        resampling_method = SystematicResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.STRATIFIED:
        resampling_method = StratifiedResampler()
    elif resampling_method_enum == ResamplingMethodsEnum.REGULARIZED:
        resampling_method = RegularisedTransform(**resampling_kwargs)
    elif resampling_method_enum == ResamplingMethodsEnum.VARIANCE_CORRECTED:
        regularized_resampler = RegularisedTransform(**resampling_kwargs)
        resampling_method = PartiallyCorrectedRegularizedTransform(regularized_resampler)
    elif resampling_method_enum == ResamplingMethodsEnum.OPTIMIZED:
        lr = resampling_kwargs.pop('lr', resampling_kwargs.pop('learning_rate', 0.1))

        loss = SinkhornLoss(**resampling_kwargs, symmetric=True)
        optimizer = SGD(loss, lr=lr, decay=0.95)
        regularized_resampler = RegularisedTransform(**resampling_kwargs)

        resampling_method = OptimizedPointCloud(optimizer, intermediate_resampler=regularized_resampler)
    elif resampling_method_enum == ResamplingMethodsEnum.CORRECTED:
        resampling_method = CorrectedRegularizedTransform(**resampling_kwargs)
    else:
        raise ValueError(f'resampling_method_name {resampling_method_enum} is not a valid ResamplingMethodsEnum')

    # np_random_state = np.random.RandomState(seed=555)

    # initial_particles = np_random_state.normal(1., 0.5, [batch_size, n_particles, M]).astype(np.float32)
    # initial_state = State(initial_particles)
    #
    # large_initial_particles = np_random_state.normal(1., 0.5, [25, n_particles, M]).astype(np.float32)
    # large_initial_state = State(large_initial_particles)
    #
    # mu_init = -5. * tf.ones(M)
    # F_init = 0.9 * tf.eye(M)
    # transition_cov_init = 0.35 * tf.eye(M)
    # observation_cov_init = 1. * tf.eye(M)
    #
    # mu = tf.Variable(mu_init, trainable=True)
    # F = tf.Variable(F_init, trainable=True)
    # transition_cov = tf.Variable(transition_cov_init, trainable=True)
    # observation_cov = tf.Variable(observation_cov_init, trainable=False)

    smc = make_filter(model, proposal, data, n_particles, resampling_method, resampling_criterion)
    surrogate_smc = make_filter(model, proposal, data, n_particles, StratifiedResampler(), resampling_criterion)

    def optimizer_maker(learning_rate):
        # tf.function doesn't like creating variables. This is a way to create them outside the graph
        # We can't reuse the same optimizer because it would be giving a warmed-up momentum to the ones run later
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        return optimizer

    #variables = [mu, F, transition_cov]
    initial_values = [v.value() for v in trainable_variables]
    initial_state = State(model.initial_state(n_particles))
    losses, ess_profiles = compile_learning_rates(smc, initial_state, observation_dataset, T, trainable_variables, initial_values,
                                                  n_iter, optimizer_maker, learning_rates, filter_seed, change_seed,
                                                  initial_state, surrogate_smc, logdir, track_gradients)

    #losses_df = pd.DataFrame(np.stack(losses).T, columns=np.log10(learning_rates))
    #ess_df = pd.DataFrame(np.stack(ess_profiles).T, columns=np.log10(learning_rates))

    #losses_df.columns.name = 'log learning rate'
    #losses_df.columns.epoch = 'epoch'

    #ess_df.columns.name = 'log learning rate'
    #ess_df.columns.epoch = 'epoch'

    # plot_losses(losses_df, resampling_method_enum.name, savefig, dx, dy, dense, T, change_seed)
    #plot_losses_vs_ess(losses_df, ess_df, resampling_method_enum.name, savefig, M, n_particles,
    #                   change_seed, batch_size, n_iter, resampling_kwargs.get("epsilon"))



FLAGS = flags.FLAGS

flags.DEFINE_integer('resampling_method', ResamplingMethodsEnum.REGULARIZED, 'resampling_method')
flags.DEFINE_float('epsilon', 0.5, 'epsilon')
flags.DEFINE_float('resampling_neff', 1., 'resampling_neff')
flags.DEFINE_float('scaling', 0.9, 'scaling')
flags.DEFINE_float('log_learning_rate_min', -2., 'log_learning_rate_min')
flags.DEFINE_float('log_learning_rate_max', -3., 'log_learning_rate_max')
flags.DEFINE_integer('n_learning_rates', 1, 'log_learning_rate_max')
flags.DEFINE_boolean('change_seed', True, 'change seed between each gradient descent step')
flags.DEFINE_float('convergence_threshold', 1e-3, 'convergence_threshold')
flags.DEFINE_integer('n_particles', 4, 'n_particles', lower_bound=4)
flags.DEFINE_integer('batch_size', 1, 'batch_size', lower_bound=1)
flags.DEFINE_integer('n_iter', 10000, 'n_iter', lower_bound=10)
flags.DEFINE_integer('max_iter', 10000, 'max_iter', lower_bound=1)
flags.DEFINE_boolean('savefig', True, 'Save fig')
flags.DEFINE_integer('seed', 222, 'seed')
flags.DEFINE_string('proposal','NormalProposal','')
flags.DEFINE_string('dataset','LinearGaussianData','')
flags.DEFINE_string('model','LinearGaussian','')
flags.DEFINE_integer('d_x',25,'')
flags.DEFINE_integer('d_y',25,'')
flags.DEFINE_string('obs','sparse','')
flags.DEFINE_string('beta_form','triangular','')
flags.DEFINE_list('lr',[1e-2,1e-3],'')
flags.DEFINE_boolean('track_gradients', False, '')

def flag_main(argb):
    for name, value in FLAGS.flag_values_dict().items():
        print(name, f'{repr(value)}')
    FLAGS.lr = [float(l) for l in FLAGS.lr]
    learning_rates = FLAGS.lr
    parameter_args = {}
    def add_arg(name, data):
        if data is not None:
            parameter_args[name]=data
    if FLAGS.dataset == 'LinearGaussianData':
        add_arg('d_x', FLAGS.d_x)
        add_arg('d_y', FLAGS.d_y)
        add_arg('obs', FLAGS.obs)
    elif FLAGS.dataset == 'StochasticVolatilityData':
        add_arg('beta_form', FLAGS.beta_form)
    proposal = getattr(sys.modules[__name__], FLAGS.proposal)
    dataset = getattr(sys.modules[__name__], FLAGS.dataset)
    model = getattr(sys.modules[__name__], FLAGS.model)
    settings = '{dataset}/dpf_{N}'.format(dataset = FLAGS.dataset, N=str(FLAGS.n_particles))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    logdir = ('final/{}/%s' % stamp).format(settings)
    writer = tf.summary.create_file_writer(logdir)
    outfile = os.path.join(logdir,'stdout')
    errfile = os.path.join(logdir,'stderr')
    sys.stdout = Logger(outfile)
    sys.stderr = Logger(errfile,sys.stderr)
    main(FLAGS.resampling_method,
         resampling_neff=FLAGS.resampling_neff,
         n_particles=FLAGS.n_particles,
         batch_size=FLAGS.batch_size,
         savefig=FLAGS.savefig,
         n_iter=FLAGS.n_iter,
         learning_rates=learning_rates,
         change_seed=FLAGS.change_seed,
         resampling_kwargs=dict(epsilon=FLAGS.epsilon,
                                scaling=FLAGS.scaling,
                                convergence_threshold=FLAGS.convergence_threshold,
                                max_iter=FLAGS.max_iter),
         filter_seed=FLAGS.seed,
         proposal = proposal,
         dataset = dataset,
         model = model,
         parameter_args = parameter_args,
         logdir=logdir,
         track_gradients=FLAGS.track_gradients)


if __name__ == '__main__':
    app.run(flag_main)
