#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=01:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
source /scratch/rp06/sl5952/Mica/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/AMBER/.cache"

cd ../..
ll .venv/lib/python3.11/site-packages/ >> uv_installed.txt
python3 -m src.run train --data-root "./datasets/ucf101_subset/temp_download/extracted/UCF101_subset" --output-dir "./runs/videomae-ucf" --epochs 4 --batch-size 4 >> T001.log 2>&1
