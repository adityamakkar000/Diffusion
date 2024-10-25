#!/bin/bash

gpu_flag=2
cpu_flag=4
mem_flag="16G"
time_flag="2:00:00"


while getopts "g:c:m:t:" flag; do
  case $flag in
    g)
      gpu_flag=$OPTARG
      ;;
    c)
      cpu_flag=$OPTARG
      ;;
    m)
      mem_flag=$OPTARG
      ;;
    t)
      time_flag=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

echo gpu ${gpu_flag}
echo cpu ${cpu_flag}
echo memory ${mem_flag}
echo time ${time_flag}

# config
SBATCH --time={time_flag}
SBATCH --mem={mem_flag}
SBATCH --cpus-per-task={cpu_flag}
SBATCH --gres=gpu:{gpu_flag}

# STD out
SBATCH -o JOB%j.out
SBATCH -e JOB%j-err.out

# email
SBATCH --mail-user=aditya.makkar000@waterloo.ca
SBATCH --mail-type=ALL

cd ~/PersonalProj/Diffusion
conda activate pytorch_base
python3 main_diffusers.py -load 
