#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --ntasks=1
#SBATCH --job-name=measure_time
#SBATCH --cpus-per-task=9
#SBATCH --time=01:00:00
#SBATCH --output=slurm_output_%A.out

source activate mulan

cd $HOME/mulan_demo
export PYTHONPATH=$HOME

METHODS=("trimap_manual" "trimap_auto" "trimap_og" "umap" "tsne")
DATASETS=("MNIST" "coil_20")

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    python measure_time.py --method "$method" --dataset "$dataset"
  done
done
