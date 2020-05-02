#!/bin/sh
##ENVIRONMENT SETTINGS;
#SBATCH --export=NONE
#SBATCH --get-user-env=L

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Resnet_RNN    
#SBATCH --time=05:00:00                    
#SBATCH --ntasks=16
#SBATCH --mem=40G             
#SBATCH --output=Resnet_CRNN.%j
#SBATCH --gres=gpu:1 

module load Anaconda3/5.0.0.1
module load CUDA
source activate /scratch/user/ssankaran30/anaconda3/envs/my_env
python videoClassification_Resnet_CRNN.py   
source deactivate
