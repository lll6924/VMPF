import tensorflow as tf
from algorithm import VariationalMarginalParticleFilter, VariationalSequentialMonteCarlo, VariationalMarginalParticleFilterBiasedGradients
from proposal import DenseNeuralNetwork, NormalProposal, PriorAndNormal
from model import StochasticVolatility, LinearGaussian, StochasticVolatilityData, LinearGaussianData
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import click
import sys
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from utils import Logger

tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)
@click.command()
@click.option('--algorithm', default='VariationalSequentialMonteCarlo')
@click.option('--proposal', default='NormalProposal')
@click.option('--samples', default = 1000)
@click.option('--dataset', default='LinearGaussianData')
@click.option('--model', default = 'LinearGaussian')
@click.option('--restore', default = 'newlogs/VariationalMarginalParticleFilter_NormalProposal_0.01_200_False_4_LinearGaussianData/20210421-183424/')
@click.option('--start', default = 4)
@click.option('--end', default = 500)
@click.option('--step', default = 4)
@click.option('--d_x', default = None)
@click.option('--d_y', default = None)
@click.option('--obs', default = None)
@click.option('--beta_form', default = None)
@click.option('--adaptive_resample', is_flag = True)
def main(algorithm, proposal, samples, dataset, model, restore, start, end, step,d_x,d_y,obs,beta_form,adaptive_resample):
    parameter_args = {}
    def add_arg(name, data):
        if data is not None:
            parameter_args[name]=data
    add_arg('d_x', d_x)
    add_arg('d_y', d_y)
    add_arg('obs', obs)
    add_arg('beta_form', beta_form)
    prop = getattr(sys.modules[__name__], proposal)
    dataset = getattr(sys.modules[__name__], dataset)
    model = getattr(sys.modules[__name__], model)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists('final/post_evaluation'):
        os.mkdir('final/post_evaluation')
    logdir = ('final/post_evaluation/%s' % stamp)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    outfile = os.path.join(logdir,'stdout')
    errfile = os.path.join(logdir,'stderr')
    sys.stdout = Logger(outfile)
    sys.stderr = Logger(errfile,sys.stderr)
    alg = getattr(sys.modules[__name__], algorithm)(proposal=prop,
                                                    ssm = model,
                                                    data = dataset,
                                                    adaptive_resample = adaptive_resample,
                                                    **parameter_args)

    variables = alg.get_trainable_variables()
    lb_data = []
    mean_lb = []
    ckpt = os.path.join(restore,'ckpt-1')
    rst = tf.train.Checkpoint()
    rst.listed = variables
    rst.restore(ckpt).assert_consumed()

    print('Restored from ',ckpt)
    for i in range(len(variables)):
        variables[i].assign(rst.listed[i])

    for n in range(start, end, step):
        print("Evaluating {} ...".format(str(n)))
        bounds = []
        for _ in tqdm(range(samples)):
            ans = alg.log_p_hat(N = n)
            bounds.append(ans.numpy())
        res = np.mean(bounds)
        print('Bound: ',res)
        lb_data.append({'n':n,'lb':bounds})
        mean_lb.append({'n':n,'lb':res})

    np.savez_compressed(os.path.join(logdir,'lb.npz'),lb=lb_data)
    mean_lb = pd.DataFrame(mean_lb)
    sns.lineplot(data=mean_lb,x='n',y='lb')
    plt.savefig(os.path.join(logdir,'lb.pdf'),format='pdf')
    plt.clf()

if __name__ == '__main__':
    main()