# Network and Deep Learning Project 1

This repository contains a NumPy-only implementation for MNIST handwritten digit classification. It implements MLP and CNN baselines plus optimization and regularization extensions required by the course project.

## What Is Implemented

- `mynn/op.py`
  - `Linear` forward/backward
  - `conv2D` forward/backward with stride and padding
  - `ReLU`
  - `Dropout`
  - softmax cross entropy
- `mynn/models.py`
  - `Model_MLP`
  - `Model_CNN`
  - `Model_CNN_Deep`
- `mynn/optimizer.py`
  - SGD
  - Momentum GD
  - RMSProp
  - Adam
- `mynn/lr_scheduler.py`
  - StepLR
  - MultiStepLR
  - ExponentialLR

## Main Results

| Model / Method | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| MLP baseline | 92.91% | 93.18% |
| CNN baseline | 95.85% | 96.05% |
| CNN + Momentum | 97.16% | 97.06% |
| CNN + Adam | 97.67% | 97.80% |
| CNN + RMSProp | 97.35% | 97.43% |
| Enhanced Deep CNN + Adam | 98.63% | 98.67% |

The final best checkpoint is `enhanced_deep_cnn_adam_best_model.pickle`, stored outside this Git repository in `../model_weight/`.

## How to Run

Run scripts from this `codes/` directory.

```bash
python train_part_a_mlp.py
python train_part_b_cnn.py
python train_part_c_experiments.py
python train_enhanced_experiments.py
```

The scripts expect the MNIST files under:

```text
dataset/MNIST/
```

The dataset is intentionally ignored by Git, following the project requirement.

## Important Project Rule

Do not upload the MNIST dataset, model checkpoints, or other large binary files to GitHub. Checkpoints are collected in:

```text
../model_weight/
```

Upload those checkpoint files to an external platform such as ModelScope, then paste the external link into the final PDF report.



