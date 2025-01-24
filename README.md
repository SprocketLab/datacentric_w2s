# Weak-to-Strong Generalization Through the Data-Centric Lens

[Changho Shin](https://ch-shin.github.io/), John Cooper, [Frederic Sala](https://pages.cs.wisc.edu/~fredsala/)

Internation Conference on Learning Representations (ICLR) 2025.

Source code for experiments, based on [EleutherAI/w2s](https://github.com/EleutherAI/w2s).

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
