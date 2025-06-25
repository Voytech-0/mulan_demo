#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=measure_time
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --output=slurm_output_%A.out

source activate mulan

cd $HOME/mulan_demo
export PYTHONPATH=$HOME

#METHODS=("trimap_pip" "trimap_manual" "trimap_auto" "trimap_og" "umap" "tsne")
#DATASETS=("MNIST" "coil_20")
METHODS=("trimap_pip" "trimap_auto" "umap")
DATASETS=("rcv1")

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    python measure_time.py --method "$method" --dataset "$dataset"
  done
done
