export CUDA_VISIBLE_DEVICES=2
python overlap_activation_save.py --seed 0 &
export CUDA_VISIBLE_DEVICES=5
python overlap_activation_save.py --seed 1 &
export CUDA_VISIBLE_DEVICES=7
python overlap_activation_save.py --seed 2 

export CUDA_VISIBLE_DEVICES=2
python overlap_activation_save.py --seed 3 &
export CUDA_VISIBLE_DEVICES=5
python overlap_activation_save.py --seed 4 &
export CUDA_VISIBLE_DEVICES=7
python overlap_activation_save.py --seed 5 

export CUDA_VISIBLE_DEVICES=2
python overlap_activation_save.py --seed 6 &
export CUDA_VISIBLE_DEVICES=5
python overlap_activation_save.py --seed 7 &
export CUDA_VISIBLE_DEVICES=7
python overlap_activation_save.py --seed 8 

export CUDA_VISIBLE_DEVICES=2
python overlap_activation_save.py --seed 9 &
