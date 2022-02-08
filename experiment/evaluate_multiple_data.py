import tensorflow as tf
from algorithm import VariationalSequentialMonteCarloMultipleData, VariationalMarginalParticleFilterMultipleData, ImportanceWeightedVariationalInferenceMultipleData, VariationalMarginalParticleFilterBiasedGradientsMultipleData
from proposal import DenseNeuralNetwork, DNN4TMC
from model import DeepMarkovModel, DeepMarkovModelData
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
from tensorflow.python.client import device_lib
import gc
tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)
@click.command()
@click.option('--algorithm', default='VariationalSequentialMonteCarloMultipleData')
@click.option('--proposal', default='DenseNeuralNetwork')
@click.option('--samples', default = 10)
@click.option('--n', default=128)
@click.option('--dataset', default='JSB')
@click.option('--d_h', default = 64)
@click.option('--restore', default = None)
@click.option('--partition', default = 'test')

def main(algorithm, proposal, samples, n, dataset, d_h, restore, partition):
    name = '{algorithm}_{dataset}_{n}_{partition}'.format(algorithm=algorithm, dataset=dataset, n=str(n),partition=partition)
    prop = getattr(sys.modules[__name__], proposal)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists('final/post_evaluation'):
        os.mkdir('final/post_evaluation')
    logdir = ('final/post_evaluation/%s' % stamp)
    writer = tf.summary.create_file_writer(logdir)
    outfile = os.path.join(logdir,'stdout')
    errfile = os.path.join(logdir,'stderr')
    sys.stdout = Logger(outfile)
    sys.stderr = Logger(errfile,sys.stderr)
    alg = getattr(sys.modules[__name__], algorithm)(proposal=prop,
                                                    data = DeepMarkovModelData(dataset=dataset),
                                                    N = n,
                                                    d_h=d_h)

    train, valid, test = alg.train, alg.valid, alg.test
    if partition == 'test':
        data = test
    elif partition == 'valid':
        data = valid
    else:
        data = train
    variables = alg.get_trainable_variables()

    ckpt = os.path.join(restore,'best-ckpt-1')

    rst = tf.train.Checkpoint()
    rst.listed = variables
    rst.restore(ckpt).assert_consumed()
    print('restored from ', ckpt)

    print("Evaluating ...")
    bounds = 0
    steps = 0
    for _ in tqdm(range(samples)):
        for y in data:
            y = tf.convert_to_tensor(y,dtype=tf.float32)
            ans = alg.log_p_hat(y)
            bounds+=ans.numpy()
            steps += len(y)
        print(bounds/steps)
    print(partition, ' lb for ',algorithm,'(',n,'): ',bounds/steps)
    res_file = os.path.join(restore,name)
    with open(res_file,'w') as f:
        print(bounds/steps,file=f)

if __name__ == '__main__':
    main()