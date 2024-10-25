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

# create submit.sh
echo \#!/bin/bash >> submit.sh

# config
echo \#SBATCH --time="$time_flag" >> submit.sh
echo \#SBATCH --mem="$mem_flag" >> submit.sh
echo \#SBATCH --cpus-per-task="$cpu_flag"> submit.sh
echo \#SBATCH --gres=gpu:"$gpu_flag" >> submit.sh

# STD out

echo \#SBATCH -o JOB%j.out >> submit.sh
echo \#SBATCH -e JOB%j-err.out >> submit.sh


# email

echo \#SBATCH --mail-user=adityamakkar000@gmail.com >> submit.sh
echo \#SBATCH --mail-type=ALL >> submit.sh


# activate env
echo source activate pytorch_base >> submit.sh
echo echo $CONDA_DEFAULT_ENV >> submit.sh

echo nvidia-smi >> submit.sh
echo cd ~/PersonalProj/Diffusion >> submit.sh
echo echo "$PWD" >> submit.sh
echo python3 main_diffusers.py -load >> submit.sh

sbatch submit.sh
rm submit.sh