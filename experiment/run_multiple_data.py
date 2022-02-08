import tensorflow as tf
from algorithm import VariationalSequentialMonteCarloMultipleData, VariationalMarginalParticleFilterMultipleData, ImportanceWeightedVariationalInferenceMultipleData, VariationalMarginalParticleFilterBiasedGradientsMultipleData
from proposal import DenseNeuralNetwork, DNN4TMC
from model import DeepMarkovModelData, DeepMarkovModel
import click
import sys
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from utils import Logger
from tensorflow.python.client import device_lib
import ast
tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)

class PythonLiteralOption(click.Option):
    # this class comes from https://stackoverflow.com/questions/47631914/how-to-pass-several-list-of-arguments-to-click-option/47730333
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

@click.command()
@click.option('--algorithm', default='VariationalSequentialMonteCarloMultipleData')
@click.option('--proposal', default='DenseNeuralNetwork')
@click.option('--lr', cls=PythonLiteralOption, default=[])
@click.option('--epochs', cls=PythonLiteralOption, default=[])
@click.option('--evaluation_n', default = 10)
@click.option('--final_evaluation_n', default = 100)
@click.option('--reparameterize', default = True)
@click.option('--n', default=4)
@click.option('--dataset', default='JSB')
@click.option('--d_h', default = 64)
@click.option('--retrieve', default = None)
@click.option('--early_stopping', is_flag = True)
@click.option('--clip_gradients', is_flag = True)
@click.option('--grad_clip_thresholds', cls=PythonLiteralOption, default='0')
def main(algorithm, proposal, lr, epochs, evaluation_n, final_evaluation_n, reparameterize, n, dataset, d_h, retrieve, early_stopping, clip_gradients, grad_clip_thresholds):
    print(tf.__version__)
    print(tf.config.list_physical_devices())
    print(device_lib.list_local_devices())
    settings = '{dataset}/{algorithm}_{proposal}_{lr}_{epochs}_{N}'.format(algorithm=algorithm,
                                                                           proposal = proposal,
                                                                            lr = str(lr),
                                                                            epochs = str(epochs),
                                                                            N=n,
                                                                            dataset=dataset,)
    if not hasattr(lr, '__iter__'):
        lr = [lr]
    if not hasattr(epochs, '__iter__'):
        epochs = [epochs]
    prop = getattr(sys.modules[__name__], proposal)
    reparameterize = str(reparameterize) == 'True'
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    logdir = ('final/{}/%s' % stamp).format(settings)
    writer = tf.summary.create_file_writer(logdir)
    outfile = os.path.join(logdir,'stdout')
    errfile = os.path.join(logdir,'stderr')
    sys.stdout = Logger(outfile)
    sys.stderr = Logger(errfile,sys.stderr)
    alg = getattr(sys.modules[__name__], algorithm)(proposal=prop,
                                                    reparameterize = reparameterize,
                                                    data = DeepMarkovModelData(dataset=dataset),
                                                    N = n,
                                                    d_h=d_h)
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    def loss(y = None):
        loss_value = alg.loss(y)
        return loss_value

    train, valid, test = alg.train, alg.valid, alg.test
    variables = alg.get_trainable_variables()

    if retrieve is not None:
        rst = tf.train.Checkpoint()
        rst.listed = variables
        rst.restore(retrieve).assert_consumed()

        print('Retrieved from ', retrieve)

    loss_data = []
    valid_data = []
    print('training points: ', len(train))
    print('validation points: ', len(valid))
    print('test points: ', len(test))
    timesteps = np.sum([len(y) for y in train])
    print('time steps: ', timesteps)
    best_val = -np.inf
    best_epoch = 0
    optimizers = [tf.optimizers.Adam(learning_rate=l) for l in lr]
    if not clip_gradients:
        grad_clip_thresholds = [0 for l in lr]
    epoch = 0
    for l, o, e, grad_clip_threshold in zip(lr, optimizers, epochs, grad_clip_thresholds):
        for _ in range(1, e+1):
            epoch += 1
            train_losses = []
            nlls = []
            max_grads = []
            np.random.shuffle(train)
            for y_train in tqdm(train):
                y_train = tf.convert_to_tensor(y_train,dtype=tf.float32)
                with tf.GradientTape() as tape:
                    nll = -alg.log_p_hat(y_train)
                    loss_value = nll
                grads = tape.gradient(loss_value, variables)
                max_grad = tf.reduce_max([tf.reduce_max(tf.abs(grad)) for grad in grads])
                if clip_gradients:
                    grads = [tf.clip_by_value(grad, -grad_clip_threshold, grad_clip_threshold) for grad in grads]
                o.apply_gradients(zip(grads, variables))
                train_losses.append(loss_value.numpy())
                max_grads.append(max_grad.numpy())
                nlls.append(nll.numpy())
            mean_loss = np.mean(train_losses)
            mean_nll = np.mean(nlls)
            print("epoch ", epoch,' train loss: ', mean_loss, 'average max grad: ', np.mean(max_grads), ' nats per step: ',-mean_nll*len(train)/timesteps, ' lr: ',l)
            loss_data.append(mean_loss)
            if epoch % 10 ==0:
                bounds = 0
                steps = 0
                for _ in tqdm(range(evaluation_n)):
                    for y_valid in valid:
                        y_valid = tf.convert_to_tensor(y_valid, dtype=tf.float32)
                        ans = alg.log_p_hat(y_valid)
                        bounds += ans.numpy()
                        steps += len(y_valid)
                val_npt = bounds/steps
                print("epoch ", epoch, ": valid npt: ",val_npt)
                valid_data.append({'epoch':epoch,'npt':val_npt})
                if val_npt > best_val:
                    saver = tf.train.Checkpoint()
                    saver.listed = variables
                    saver.save(os.path.join(logdir, 'best-ckpt'))
                    print("Saved the best ckpt at epoch ",epoch)
                    best_val = val_npt
                    best_epoch = epoch
            if epoch % 100 == 0:
                saver = tf.train.Checkpoint()
                saver.listed = variables
                saver.save(os.path.join(logdir, 'middle-ckpt'))
                np.savez_compressed(os.path.join(logdir, 'loss.npz'), loss=loss_data, valid_data=valid_data)
            if early_stopping and epoch - best_epoch >= 50:
                print('training stopped due to early stopping')
                break

    saver = tf.train.Checkpoint()
    saver.listed = variables
    saver.save(os.path.join(logdir,'final-ckpt'))
    np.savez_compressed(os.path.join(logdir,'loss.npz'),loss=loss_data,valid_data=valid_data)


if __name__ == '__main__':
    main()