#!/bin/bash
#SBATCH --job-name=ref-nerf                  # create a short name for your job
#SBATCH --nodes=1                            # node count
#SBATCH --ntasks=1                           # total number of tasks across all nodes
#SBATCH --cpus-per-task=4                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2                         # number of gpus per node
#SBATCH --time=01:30:00                      # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin                    # send email when job begins
#SBATCH --mail-type=end                      # send email when job ends
#SBATCH --mail-user=smohr@princeton.edu      # address to send emails to
#SBATCH --output=slurm_scripts/slurm-%j.out  # where to store the output file

module purge
module load anaconda3/2022.5 cudatoolkit/11.7 cudnn/cuda-11.5/8.3.2
conda activate /scratch/network/smohr/multinerf-env

nvidia-smi

ScratchDir="/tmp/ref-nerf"
mkdir -p ${ScratchDir}

export CUDA_VISIBLE_DEVICES=0,1

SCENE=toaster_transformer
EXPERIMENT=shinyblender
DATA_DIR="/tmp/ref-nerf/data"  # create a name for the directory
mkdir -p ${DATA_DIR}           # make the directory
cp -r /scratch/network/smohr/cos526/multinerf/data/refnerf/"${SCENE}" ${DATA_DIR}
CHECKPOINT_DIR=/scratch/network/smohr/cos526/multinerf/checkpoints/"${EXPERIMENT}"/"${SCENE}"
mkdir -p ${CHECKPOINT_DIR}
rm "$CHECKPOINT_DIR"/*

python -m train \
  --gin_configs=configs/blender_refnerf_small.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr

python -m eval \
  --gin_configs=configs/blender_refnerf_small.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr

python -m render \
  --gin_configs=configs/blender_refnerf_small.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_dir = '${CHECKPOINT_DIR}/render/'" \
  --gin_bindings="Config.render_video_fps = 15" \
  --logtostderr

# SCENE=ball
# cp -r /scratch/network/smohr/cos526/multinerf/data/refnerf/"${SCENE}" ${DATA_DIR}
# CHECKPOINT_DIR=/scratch/network/smohr/cos526/multinerf/checkpoints/"${EXPERIMENT}"/"${SCENE}"
# mkdir -p ${CHECKPOINT_DIR}
# rm "$CHECKPOINT_DIR"/*

# python -m train \
#   --gin_configs=configs/blender_refnerf.gin \
#   --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
#   --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
#   --logtostderr
