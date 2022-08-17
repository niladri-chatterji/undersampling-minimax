# Undersampling is a Minimax Optimal Robustness Intervention in Nonparametric Classification

![Polynomially-tailed loss correcting for distribution shift](media/linear_intuition.jpg)
```
@article{chatterji2022undersampling,
  title={Undersampling is a Minimax Optimal Robustness Intervention in Nonparametric Classification},
  author={Chatterji, Niladri S and Haque, Saminul and Hashimoto, Tatsunori},
  journal={arXiv preprint arXiv:2205.13094},
  year={2022}
}
```

- [Code](https://github.com/niladri-chatterji/undersampling-minimax)
- [arXiv](https://arxiv.org/abs/2205.13094)

This repo is created by from the repository [importance-weighting-interpolating-classifiers](https://github.com/KeAWang/importance-weighting-interpolating-classifiers). It uses the template from [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
Experiment settings are managed by [hydra](https://hydra.cc/), a hierarchical config framework, and the setting files are specified in the `configs/` directory.

## Setup instructions

0. Make a weights and bias account
1. `conda env create -f conda_env.yml`
2. `pip install requirements.txt`
3. Copy `.env.tmp` to a new file called `.env`. Edit the `PERSONAL_DIR` environment variable in `.env` to be the root directory of where you want to store your data.

## Reproducing experiments

The commandline scripts below will run with the default seed. In our paper we loop over seeds for each experiment, which you can do by appending `seed=0,1,2,3,4` to the launch script below.

Some seeds may result in NaNs during training. Relaunching the experiments (without changing the seed) will get rid of the NaNs, likely due to GPU non-determinism.

### Figure 1

Run `notebooks/two-gaussians.ipynb`

### Figure 2 (Importance Weighted Cross-Entropy Loss and VS Loss)


```bash
# Importance Weighted Cross-Entropy Loss Minority Samples Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[2500,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[2500,1000]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[2500,1500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[2500,2000]

# Importance Weighted Cross-Entropy Loss Majority Samples Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[3000,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[3500,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[4000,500]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[4500,500]

# Importance Weighted Cross-Entropy Loss Propotional Increase Sweep
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[3000,600]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[3500,700]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[4000,800]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[4500,900]
python run.py +experiment=cifar_reweighted_early_stopped_scaling trainer.max_epochs=800 datamodule.class_samples=[5000,1000]

# Group DRO Minority Samples Sweep
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,500]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,1000]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,1500]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,2000]

# Group DRO Majority Samples Sweep
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3000,500]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3500,500]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4000,500]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4500,500]

# Group DRO Propotional Increase Sweep
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3000,600]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3500,700]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4000,800]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4500,900]
python run.py +experiment=cifar_early_stopped_scaling_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[5000,1000]
```

### Figure 3 (Hat Function)

Run `notebooks/two-gaussians.ipynb`


### Figure 4 (Tilted loss and Group DRO)

```bash
# Tilted Loss Minority Samples Sweep
python run.py +experiment=cifar_scaling_minority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,500]
python run.py +experiment=cifar_scaling_minority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,1000]
python run.py +experiment=cifar_scaling_minority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,1500]
python run.py +experiment=cifar_scaling_minority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[2500,2000]

# Tilted Loss Majority Samples Sweep
python run.py +experiment=cifar_scaling_majority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3000,500]
python run.py +experiment=cifar_scaling_majority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3500,500]
python run.py +experiment=cifar_scaling_majority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4000,500]
python run.py +experiment=cifar_scaling_majority_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4500,500]

# Tilted Loss Propotional Increase Sweep
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3000,600]
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[3500,700]
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4000,800]
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[4500,900]
python run.py +experiment=cifar_scaling_n_tiltedloss trainer.max_epochs=800 datamodule.class_samples=[5000,1000]

# Group DRO Minority Samples Sweep
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[2500,500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[2500,1000]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[2500,1500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[2500,2000]

# Group DRO Majority Samples Sweep
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[3000,500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[3500,500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[4000,500]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[4500,500]

# Group DRO Propotional Increase Sweep
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[3000,600]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[3500,700]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[4000,800]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[4500,900]
python run.py +experiment=cifar_early_stopped_scaling_dro trainer.max_epochs=800 datamodule.class_samples=[5000,1000]
```
