#!/bin/bash
#SBATCH -J pretext # job name
#SBATCH -o log_slurm.o%j # output and error file name (%j expands to jobID)
#SBATCH -p gpu # queue (partition) -- defq, eduq, gpuq, shortq
#SBATCH -t 64:00:00 # run time (hh:mm:ss) - 12.0 hours in this example.

module load slurm
module load cuda10.0/toolkit/10.0.130 
module load gcc


# Execute the program

python simclr.py --config_env configs/env.yml --config_exp configs/pretext/moco_rico.yml
#python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_stl10.yml
#python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml


#python eval.py --config_exp configs/scan/scan_cifar10.yml --model /bsuhome/hkiesecker/scratch/imageClassification/US/cifar-10/scan_cifar-10.pth.tar --visualize_prototypes