#!/bin/bash
#SBATCH -J pretext # job name
#SBATCH -o log_slurm.o%j # output and error file name (%j expands to jobID)
#SBATCH -p gpu # queue (partition) -- defq, eduq, gpuq, shortq
#SBATCH -t 12:00:00 # run time (hh:mm:ss) - 12.0 hours in this example.

module load slurm
module load cuda10.0/toolkit/10.0.130 
module load gcc


# Execute the program
#python fixing_dataloader_step1Scan.py
#python image_2Bin.py
#python /bsuhome/hkiesecker/scratch/imageClassification/US/rico_20/evaluate_bin.py


### we are begining to debug from this point in a remote cluster, a run causing a run in another system
### should theoretically be able to run this on a remote connection of a remote cluster the same way as just a remote conneciton
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_stl10.yml
#python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml
#python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml


#python eval.py --config_exp configs/scan/scan_cifar10.yml --model /bsuhome/hkiesecker/scratch/imageClassification/US/cifar-10/scan_cifar-10.pth.tar --visualize_prototypes