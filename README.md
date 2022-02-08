# Variational Marginal Particle Filters

## Background

Tensorflow implementation for AISTATS 2022 paper: Variational Marginal Particle Filters.

The differentiable particle filtering part of the codes is modified from https://github.com/JTT94/filterflow. 

## Dependencies

* Python 3.6.8
* Tensorflow 2.4.1
* Tensorflow-Probability 0.12.2

See `requirements.txt` for details.


## Training Commands

### Linear Gaussian State Space Models

For example, if we want to reproduce the result of Linear Gaussian SSMs with ($d_x=25$, $d_y=25$, C sparse) with $N=4$. We need to run

```bash
python -m experiment.run --d_x 25 --d_y 25 --obs sparse --n 4 --lr 0.01,0.001 --epochs 10000,10000 --algorithm 'algorithm'
```
where 'algorithm' is 
* IWVI: `ImportanceWeightedVariationalInference`
* VSMC: `VariationalSequentialMonteCarlo`
* VMPF-BG: `VariationalMarginalParticleFilterBiasedGradients`
* VMPF-UG: `VariationalMarginalParticleFilter`

The saved model and training procedure is saved at `final/LinearGaussianData/`. If we want to retrieve the parameters of some saved model and continue training, the option `--retrieve  final/LinearGaussianData/.../ckpt-1` should be added. 

To train TMC for this model and stochastic volatility models, additional option `--proposal Normal4TMC` should be added to the training scripts for VMPF-UG. For DPF, run `python -m experiment.differentiable_particle_filter --n_particles 4` for the same comparison.

### Stochastic Volatility Models

We use the same training scripts as Linear Gaussian SSM, except that additional options `--proposal PriorAndNormal --dataset StochasticVolatilityData --model StochasticVolatility` should be included to specify the model, proposal and dataset.

If we want to run VMPF-BG on triangular B, we need

```bash
python -m experiment.run --proposal PriorAndNormal --dataset StochasticVolatilityData --model StochasticVolatility --beta_form triangular --n 4 --lr 0.003,0.0003,0.00003,0.000003 --epochs 100000,100000,100000,100000 --algorithm VariationalMarginalParticleFilterBiasedGradients
```

For diagonal B, we replace 'triangular' with 'diagonal' and adjust the learning rates as well as epochs.

### Deep Markov Models

Deep Markov models use multiple data instead of single data. So we use a different training code. If we want to train on JSB dataset with VMPF-BG, the following command is needed.

```bash
python -m experiment.run_multiple_data --lr 0.001,0.0001 --n 4 --dataset JSB --epochs 1000,200 --d_h 64 --algorithm VariationalMarginalParticleFilterBiasedGradientsMultipleData
```

Note that the name of all algorithms are added with 'MultipleData' to distinguish from single data case. And we can replace the dataset name with 'Piano-midi.de', 'Nottingham' or 'MuseData' to run on the other three datasets.

To train TMC, additional option `--proposal DNN4TMC` should be added to the training scripts for VMPF-UG. For DPF, run `python -m experiment.dpf_multiple_data --lr 0.001,0.0001 --n_particles 4 --dataset JSB --n_iter 1000,200 --d_h 64` for the same comparison.