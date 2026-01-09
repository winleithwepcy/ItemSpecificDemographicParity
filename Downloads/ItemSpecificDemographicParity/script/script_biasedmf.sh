#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g4_dev
#$ -ac d=nvcr-pytorch-2312

# Load environment
source ~/.bash_profile

# Change to working directory
cd /home/win/ISDP_ML_Ours || { echo "Error: Failed to change to /home/win/RF"; exit 1; }

#python main.py --model FairBiasedMF --lr 0.0006612453458715559 --latent_dim 128 --batch_size 128 --seed 2040 --epochs 20 --num_negatives 4 --top_k 10 --fair_lambda 0.0
python main.py --model FairBiasedMF --lr 0.0006612453458715559 --latent_dim 128 --batch_size 128 --seed 2040 --epochs 20 --num_negatives 4 --top_k 10 --fair_lambda 0.2
#python main.py --model FairBiasedMF --lr 0.0006612453458715559 --latent_dim 128 --batch_size 128 --seed 2040 --epochs 20 --num_negatives 4 --top_k 10 --fair_lambda 0.4
#python main.py --model FairBiasedMF --lr 0.0006612453458715559 --latent_dim 128 --batch_size 128 --seed 2040 --epochs 20 --num_negatives 4 --top_k 10 --fair_lambda 0.6
#python main.py --model FairBiasedMF --lr 0.0006612453458715559 --latent_dim 128 --batch_size 128 --seed 2040 --epochs 20 --num_negatives 4 --top_k 10 --fair_lambda 0.8
#python main.py --model FairBiasedMF --lr 0.0006612453458715559 --latent_dim 128 --batch_size 128 --seed 2040 --epochs 20 --num_negatives 4 --top_k 10 --fair_lambda 1.0