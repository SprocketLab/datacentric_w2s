# Weak-to-Strong Generalization Through the Data-Centric Lens

[Changho Shin](https://ch-shin.github.io/), John Cooper, [Frederic Sala](https://pages.cs.wisc.edu/~fredsala/)

Internation Conference on Learning Representations (ICLR) 2025.

Paper Link: [https://arxiv.org/abs/2404.08461
](https://arxiv.org/abs/2412.03881)

![image](https://github.com/user-attachments/assets/1824000e-6613-48a2-8ec2-205ff3dfef5b)

## Abstract
The weak-to-strong generalization phenomenon is the driver for important machine learning applications including highly data-efficient learning and, most recently, performing superalignment. While decades of research have resulted in numerous algorithms that produce strong empirical performance, understanding what aspects of data enable weak-to-strong generalization has been understudied. We propose a simple data-centric mechanism that characterizes weak-to-strong generalization: the overlap density. Intuitively, generalization tracks the number of points that contain overlaps, i.e., both easy patterns (learnable by a weak model) and challenging patterns (only learnable by a stronger model), as with such points, weak predictions can be used to learn challenging patterns by stronger models. And, we provide a practical overlap detection algorithm to find overlap density from data. Finally, we provide an algorithm to learn, among multiple sources of data, which to query when seeking to maximize overlap density and thereby enhance weak-to-strong generalization. We provide a theoretical result showing that the generalization benefit is a function of the overlap density and a regret bound of our data selection algorithm. Empirically, we validate the mechanism and the overlap detection algorithm on a wide array of settings.


## Installation

`pip install -e .`

If you run into problems, try installing inside a conda or venv environment.

## Running experiments

* Overlap detection experiments
  
Step 1. [Cache activations](https://github.com/SprocketLab/datacentric_w2s/blob/main/scripts/llm_probing_overlap_detection/overlap_activation_save.py)

Step 2. [Run overlap detection experiments](https://github.com/SprocketLab/datacentric_w2s/blob/main/scripts/llm_probing_overlap_detection/overlap_probing_experiment.py)

* [Data source selection experiments](https://github.com/SprocketLab/datacentric_w2s/blob/main/scripts/llm_probing_data_selection/data_selection_with_linear_probing.py)

## Citation
```tex
@article{shin2024weak,
  title={Weak-to-Strong Generalization Through the Data-Centric Lens},
  author={Shin, Changho and Cooper, John and Sala, Frederic},
  journal={arXiv preprint arXiv:2412.03881},
  year={2024}
}
```
Acknowledgement: This source code is built upon [EleutherAI's W2S repository](https://github.com/EleutherAI/w2s).
