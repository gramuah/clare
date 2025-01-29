#!/bin/bash


#CIFAR10
python CLARE.py --dataset CIFAR10 --num_classes_per_task 2 --num_tasks 5 --seed 1000 --memory_size 1000 --num_passes 500 --model CLIP  --scenario unrealistic  --exp_name CIFAR10_unrealistic

python CLARE.py --dataset CIFAR10 --num_tasks 5 --seed 1000 --memory_size 1000 --num_passes 500 --model CLIP  --scenario semirealistic  --exp_name CIFAR10_semirealistic

python CLARE.py --dataset CIFAR10 --num_tasks 5 --seed 1000 --memory_size 1000 --num_passes 500 --model CLIP  --scenario realistic  --exp_name CIFAR10_realistic


#CIFAR100
python CLARE.py --dataset CIFAR100 --num_classes_per_task 5 --num_tasks 20 --seed 1000 --memory_size 2000 --num_passes 500 --model CLIP  --scenario unrealistic  --exp_name CIFAR100_unrealistic

python CLARE.py --dataset CIFAR100 --num_tasks 20 --seed 1000 --memory_size 2000 --num_passes 500 --model CLIP  --scenario semirealistic  --exp_name CIFAR100_semirealistic

python CLARE.py --dataset CIFAR100 --num_tasks 20 --seed 1000 --memory_size 2000 --num_passes 500 --model CLIP  --scenario realistic  --exp_name CIFAR100_realistic


