#! /bin/sh

#SBATCH --job-name=bertscore_gpt5prompt
#SBATCH --partition gpu-a100-q
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output bertscore-%j.out
#SBATCH --error bertscore-%j.err

source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate base
conda activate /home/common/ACNLP/conda_envs/eng2nav

python 5_prompt.py
