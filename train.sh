#!/bin/bash

#SBATCH --job-name=lr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=96:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100

# job info
# LOSS=$1

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
--overlay /scratch/lg154/sseg/dataset/tiny-imagenet-200.sqf:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python train.py --batch_size 512 --ufm --exp_name ufm_b512_ob_debug
 " 


 # --bias 