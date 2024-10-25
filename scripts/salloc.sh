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

salloc --gres=gpu:${gpu_flag} --cpus-per-task=${cpu_flag} --mem=${mem_flag} --time=${time_flag}
