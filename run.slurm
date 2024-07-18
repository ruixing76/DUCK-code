#! /bin/bash
#SBATCH --job-name=duck # Job name
#SBATCH --output=sbatch_output/slurm.%N.%j.out # Standard output and error.
#SBATCH --mem=32G # Total RAM to be used
#SBATCH --time=24:00:00 # Specify the time needed for your experiment
#SBATCH --nodelist=ws-l6-020 # use RTX 4090


# cd ~/data/rui_codes/emo_fc/
source activate duck

sh scripts/run_ccct.sh

# wait for debug
# tail -f /dev/null