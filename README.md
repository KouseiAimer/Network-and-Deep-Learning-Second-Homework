# 神经网络与深度学习第二次作业代码

本仓库是课程 Project 1 的代码提交部分，任务为基于 MNIST 的手写数字分类。代码主要使用 NumPy 实现，不依赖 PyTorch/TensorFlow 等深度学习框架，包含 MLP baseline、CNN baseline，以及 Part C 中选择的 Optimization 和 Regularization 扩展实验。

## 项目链接

- GitHub 仓库：https://github.com/KouseiAimer/Network-and-Deep-Learning-Second-Homework
- ModelScope 权重：https://www.modelscope.cn/models/KouseiAimer/Network-and-Deep-Learning-Second-Homework
- 最终报告：`../Project/project.pdf`

## 已实现内容

- `mynn/op.py`
  - `Linear` 前向传播与反向传播
  - `conv2D` 前向传播与反向传播，支持 stride 和 padding
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

## 主要结果

| 模型 / 方法 | 验证集准确率 | 测试集准确率 |
|---|---:|---:|
| MLP baseline | 92.91% | 93.18% |
| CNN baseline | 95.85% | 96.05% |
| CNN + Momentum | 97.16% | 97.06% |
| CNN + Adam | 97.67% | 97.80% |
| CNN + RMSProp | 97.35% | 97.43% |
| Enhanced Deep CNN + Adam | 98.63% | 98.67% |

最终最优模型为 `Enhanced Deep CNN + Adam`，对应 checkpoint 文件为：

```text
enhanced_deep_cnn_adam_best_model.pickle
```

该权重未上传到 GitHub，而是按照作业要求上传到了 ModelScope。

## 如何运行

请在当前 `codes/` 目录下运行脚本：

```bash
python train_part_a_mlp.py
python train_part_b_cnn.py
python train_part_c_experiments.py
python train_enhanced_experiments.py
```

脚本默认读取 MNIST 数据集路径：

```text
dataset/MNIST/
```

数据集文件不包含在 GitHub 仓库中，需要按照课程提供的 starter code 目录结构自行放置。

## 文件说明

- `train_part_a_mlp.py`：训练 Part A 的 MLP baseline。
- `train_part_b_cnn.py`：训练 Part B 的 CNN baseline，并生成学习曲线与卷积核可视化。
- `train_part_c_experiments.py`：运行 Part C 的初始优化器与 L2 正则化对比实验。
- `train_enhanced_experiments.py`：运行扩展实验，包括 Adam、RMSProp、Dropout、Early Stopping 和更深 CNN。
- `test_train.py`、`test_model.py`：保留 starter code 风格的训练与测试入口。
- `weight_visualization.py`：权重可视化辅助脚本。

## 提交说明

根据 `project_1.pdf` 要求，GitHub 仓库不包含：

- MNIST 数据集；
- 训练好的模型权重；
- notebook 缓存、图片输出、LaTeX 编译缓存等大文件或生成文件。

模型权重集中存放在本地 `../model_weight/`，并已上传至 ModelScope：

```text
https://www.modelscope.cn/models/KouseiAimer/Network-and-Deep-Learning-Second-Homework
```
