export CUDA_VISIBLE_DEVICES=0
python overlap_probing_experiment.py --seed 5 &
python overlap_probing_experiment.py --seed 6 &
python overlap_probing_experiment.py --seed 7 &
python overlap_probing_experiment.py --seed 8 &
python overlap_probing_experiment.py --seed 9
