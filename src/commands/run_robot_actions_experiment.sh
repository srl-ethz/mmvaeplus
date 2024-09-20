#!/bin/bash

OUTPUTDIR="../outputs"
EXPERIMENT="RobotActions_1"
DATADIR="../data"
EPOCHS=100
SEED=2
SHARED_LAT_DIM=32
MS_LAT_DIM=32
MLP_HIDDEN_DIM=1024
NUM_HIDDEN_LAYERS=2
# Train MMVAEplus

python train_MMVAEplus_robot_actions.py --experiment $EXPERIMENT --obj "elbo" --K 1 --batch-size 128 --epochs $EPOCHS \
      --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta 2.5 \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --priorposterior "Normal" --mlp_hidden_dim $MLP_HIDDEN_DIM --num_hidden_layers $NUM_HIDDEN_LAYERS

