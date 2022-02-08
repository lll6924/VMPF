import tensorflow as tf
from algorithm import VariationalMarginalParticleFilter, VariationalSequentialMonteCarlo, VariationalMarginalParticleFilterBiasedGradients, ImportanceWeightedVariationalInference
from proposal import DenseNeuralNetwork, NormalProposal, PriorAndNormal, Normal4TMC
from model import StochasticVolatility, LinearGaussian, StochasticVolatilityData, LinearGaussianData, StochasticVolatilityParameters
import time
import click
import sys
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from utils import Logger
import ast

class PythonLiteralOption(click.Option):
    # this class comes from https://stackoverflow.com/questions/47631914/how-to-pass-several-list-of-arguments-to-click-option/47730333
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)
@click.command()
@click.option('--algorithm', default='VariationalMarginalParticleFilter')
@click.option('--proposal', default='NormalProposal')
@click.option('--lr', cls=PythonLiteralOption, default=[])
@click.option('--epochs', cls=PythonLiteralOption, default=[])
@click.option('--evaluation_n', default = 100)
@click.option('--final_evaluation_n', default = 1000)
@click.option('--reparameterize/--no-reparameterize', default = True)
@click.option('--n', default=4)
@click.option('--dataset', default='LinearGaussianData')
@click.option('--model', default = 'LinearGaussian')
@click.option('--track_gradients', is_flag = True)
@click.option('--adaptive_resample', is_flag = True)
@click.option('--retrieve', default = None)
@click.option('--d_x', default = None)
@click.option('--d_y', default = None)
@click.option('--obs', default = None)
@click.option('--beta_form', default = None)
@click.option('--iaf_hidden_units', default=None)
@click.option('--num_iafs', default=None)
@click.option('--clip_gradients', is_flag = True)
@click.option('--grad_clip_threshold', default=10.)

def main(algorithm, proposal, lr, epochs, evaluation_n, final_evaluation_n, reparameterize, n, dataset, model, track_gradients, adaptive_resample, retrieve, d_x, d_y, obs, beta_form, iaf_hidden_units, num_iafs, clip_gradients, grad_clip_threshold):
    settings = '{dataset}/{algorithm}_{proposal}_{lr}_{epochs}_{N}'.format(algorithm=algorithm,
                                                                        proposal = proposal,
                                                                        lr = str(lr),
                                                                        epochs = str(np.sum(epochs)),
                                                                        N=n,
                                                                        dataset=dataset)
    if not hasattr(lr, '__iter__'):
        lr = [lr]
    if not hasattr(epochs, '__iter__'):
        epochs = [epochs]
    if adaptive_resample:
        settings += '_AdaptiveResampling'
    if track_gradients:
        settings += '_GradientsTracked'
    parameter_args = {}
    def add_arg(name, data):
        if data is not None:
            parameter_args[name]=data
    add_arg('d_x', d_x)
    add_arg('d_y', d_y)
    add_arg('obs', obs)
    add_arg('beta_form', beta_form)
    add_arg('hidden_units', iaf_hidden_units)
    add_arg('num_iafs', num_iafs)

    prop = getattr(sys.modules[__name__], proposal)
    dataset = getattr(sys.modules[__name__], dataset)
    model = getattr(sys.modules[__name__], model)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    print('stamp: ', stamp)
    logdir = ('final/{}/%s' % stamp).format(settings)
    writer = tf.summary.create_file_writer(logdir)
    outfile = os.path.join(logdir,'stdout')
    errfile = os.path.join(logdir,'stderr')
    sys.stdout = Logger(outfile)
    sys.stderr = Logger(errfile,sys.stderr)
    #tf.summary.trace_on(graph=True, profiler=True)
    alg = getattr(sys.modules[__name__], algorithm)(proposal=prop,
                                                    reparameterize = reparameterize,
                                                    adaptive_resample = adaptive_resample,
                                                    ssm = model,
                                                    data = dataset,
                                                    N = n,
                                                    **parameter_args)

    optimizers = [tf.optimizers.Adam(learning_rate=l) for l in lr]

    def loss():
        loss_value = alg.loss() #+ tf.compat.v1.losses.get_regularization_loss()
        return loss_value

    variables = alg.get_trainable_variables()

    if retrieve is not None:
        rst = tf.train.Checkpoint()
        rst.listed = variables
        rst.restore(retrieve).assert_consumed()

        print('Retrieved from ', retrieve)

    loss_data = []
    grad_var_data = []
    variable_number = np.sum([np.prod(v.shape) for v in variables])
    print('number of parameters: ',variable_number)
    last_time = time.time()
    iteration = 0
    for l, o, e in zip(lr, optimizers, epochs):
        for epoch in range(1, e + 1):
            iteration += 1
            with tf.GradientTape() as tape:
                loss_value = loss()
            grads = tape.gradient(loss_value, variables)
            max_grad = tf.reduce_max([tf.reduce_max(tf.abs(grad)) for grad in grads])
            if clip_gradients:
                grads = [tf.clip_by_value(grad, -grad_clip_threshold, grad_clip_threshold) for grad in grads]
            o.apply_gradients(zip(grads, variables))
            if epoch % 100 ==0:
                bounds = []
                for _ in range(evaluation_n):
                    ans = alg.log_p_hat()
                    bounds.append(ans.numpy())
                current_time = time.time()
                print("iteration ", iteration, ": grads: ", max_grad.numpy(), " lb: ",np.mean(bounds), ' std: ',np.std(bounds), ' lr: ',l, ' time period: ', current_time - last_time)
                last_time = current_time
                loss_data.append({'iteration':iteration,'lb':np.mean(bounds)})
                if track_gradients:
                    gradients_sum = np.zeros(variable_number)
                    gradients_squared_sum = np.zeros(variable_number)
                    for _ in range(evaluation_n):
                        with tf.GradientTape() as tape:
                            loss_value = loss()
                        grads = tape.gradient(loss_value,variables)
                        flattened_grads = np.concatenate([g.numpy().flatten() for g in grads])
                        gradients_sum += flattened_grads
                        gradients_squared_sum += np.square(flattened_grads)
                    gradients_mean = gradients_sum / evaluation_n
                    gradients_squared_mean = gradients_squared_sum / evaluation_n
                    gradients_var = gradients_squared_mean - np.square(gradients_mean)
                    mean_var = np.mean(gradients_var)
                    print('log mean var of gradients: ', np.log(mean_var))
                    grad_var_data.append({'iteration':iteration, 'mean_var':mean_var})
            if iteration % 10000 == 0:
                saver = tf.train.Checkpoint()
                saver.listed = variables
                saver.save(os.path.join(logdir, 'last-ckpt'))
                np.savez_compressed(os.path.join(logdir, 'loss.npz'), loss=loss_data, grad_var_data = grad_var_data)


    saver = tf.train.Checkpoint()
    saver.listed = variables
    saver.save(os.path.join(logdir,'ckpt'))

    print("Evaluating ...")
    bounds = []
    for _ in tqdm(range(final_evaluation_n)):
        ans = alg.log_p_hat()
        bounds.append(ans.numpy())
    print('Bound for ',algorithm,': ',np.mean(bounds))
    np.savez_compressed(os.path.join(logdir,'loss.npz'),loss=loss_data, grad_var_data = grad_var_data)


if __name__ == '__main__':
    main()