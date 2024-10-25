#!/bin/bash


#SBATCH --time=0:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# STD out
#SBATCH -o JOB%j.out
#SBATCH -e JOB%j-err.out

# email
#SBATCH --mail-user=aditya.makkar000@waterloo.ca
#SBATCH --mail-type=ALL

nvidia-smi
cd ~/PersonalProj/Diffusion
echo "$PWD"
source activate pytorch_base
python3 main_diffusers.py -load
