#!/bin/bash

OUTPUTDIR="./outputs"
EXPERIMENT="RobotActions_1"
DATADIR="/mnt/data/erbauer/retargeting/retargeted_hand_dataset_grab_v3.npy"
EPOCHS=130
SEED=2
SHARED_LAT_DIM=64
MS_LAT_DIM=64

FAIVE_ENC_HIDDEN_DIM=32
FAIVE_DEC_HIDDEN_DIM=32
MANO_ENC_HIDDEN_DIM=128
MANO_DEC_HIDDEN_DIM=128
GRIPPER_ENC_HIDDEN_DIM=4
GRIPPER_DEC_HIDDEN_DIM=4
NUM_HIDDEN_LAYERS=1
CUDA_DEVICE_ID=4
# Train MMVAEplus

python mmvaeplus/train_MMVAEplus_robot_actions.py --experiment $EXPERIMENT --obj "elbo" --K 1 --batch-size 16384 --epochs $EPOCHS \
      --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta 2.5 \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --priorposterior "Normal"  --num_hidden_layers $NUM_HIDDEN_LAYERS \
      --faive_enc_hidden_dim $FAIVE_ENC_HIDDEN_DIM --faive_dec_hidden_dim $FAIVE_DEC_HIDDEN_DIM \
      --mano_enc_hidden_dim $MANO_ENC_HIDDEN_DIM --mano_dec_hidden_dim $MANO_DEC_HIDDEN_DIM \
      --gripper_enc_hidden_dim $GRIPPER_ENC_HIDDEN_DIM --gripper_dec_hidden_dim $GRIPPER_DEC_HIDDEN_DIM \
      --cuda-device-id $CUDA_DEVICE_ID

