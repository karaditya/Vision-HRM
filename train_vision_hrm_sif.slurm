#!/bin/bash -l
#SBATCH --job-name=vision-hrm-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/akar/PRIVE/logs/%x_%j.out
#SBATCH --error=/home/akar/PRIVE/logs/%x_%j.err

:
# ----------------------------
# Directories
# ----------------------------

DIR_ADI=$HOME/adi_jobs/Vision-HRM
DATA_DIR=$DIR_ADI/data
LOG_DIR=$DIR_ADI/logs
SIF_PATH=/apps/containerCollections/CUDA12/pytorch-NGC-25-01.sif   # Path to CUDA12 container

mkdir -p $DATA_DIR $LOG_DIR

# ----------------------------
# WandB authentication (optional)
# ----------------------------

export WANDB_API_KEY="eae3956ca6a70efb302b0b501d3df086e86a3a6d"

# ----------------------------
# Build dataset (if needed)
# ----------------------------

if [ ! -d "$DATA_DIR/CIFAR10-processed" ]; then
    echo "Building CIFAR10 dataset..."
    singularity exec --nv \
        --bind $DIR_ADI:$DIR_ADI \
        $SIF_PATH \
        python $DIR_ADI/dataset/build_cifar_dataset.py \
        --output-dir $DATA_DIR/CIFAR10-processed --num-aug 4
fi

# ----------------------------
# Run training inside Singularity
# ----------------------------

echo "Starting Vision-HRM training..."

singularity exec --nv \
    --bind $DIR_ADI:$DIR_ADI \
    --env OMP_NUM_THREADS=8,HYDRA_FULL_ERROR=1,WANDB_API_KEY=$WANDB_API_KEY \
    $SIF_PATH \
    bash -c "
        cd $DIR_ADI || exit 1
        python pretrain_vision.py --config-path $DIR_ADI/config --config-name cfg_vision_pretrain
    "

echo "Training complete."
