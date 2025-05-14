#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --ntasks=8
#SBATCH --mem=50g
#SBATCH --tmp=500g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rajal038@umn.edu
#SBATCH --gres=gpu:1
rm -rf Qwen2-VL-Finetune
# clone data/script repo:
git clone https://github.com/majaraja/Qwen2-VL-Finetune.git

# Get container from dockerhub
apptainer pull vlm.sif docker://john119/vlm:latest

apptainer exec --nv vlm.sif bash -c "
  source activate train &&
  cd Qwen2-VL-Finetune &&
  bash scripts/finetune_lora.sh
"
