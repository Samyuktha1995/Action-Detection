#!/bin/sh
##ENVIRONMENT SETTINGS;
#SBATCH --export=NONE
#SBATCH --get-user-env=L

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Job_trial1     
#SBATCH --time=01:00:00                    
#SBATCH --ntasks=1
#SBATCH --mem=10G             
#SBATCH --output=3DCNN.%j
#SBATCH --gres=gpu:1 

module load Anaconda3/5.0.0.1
module load CUDA
source activate /scratch/user/ssankaran30/anaconda3/envs/my_env
python videoClassification.py   
source deactivate
