Shell Script

#!/bin/bash
#SBATCH --job-name=a2c_job
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=a2c_output_%j.txt
#SBATCH --error=a2c_error_%j.txt
module load python/3.7.4
python a2c.py
