# csci5527-final-code

Various code used for the our course project

Contents include

- `5527-Project`, a repository with: final docker stuff, some data generation, prompt generation
- `Qwen2-VL-Finetune`, a repository that was forked, modified, and extended for our use case
- `ros2_ws`, a repository used to generate a ros2 docker environment that can be run locally or on MSI.
- `data_wrangling.ipynb`, pretty much what it sounds like
- `finetune_qwen.slurm`, simply run this job on UMN MSI, with 'sbatch -p msigpu --gres gpu:a100:1 finetune_qwen.slurm' to finetune Qwen2.5 VL 7b. Thats all.
