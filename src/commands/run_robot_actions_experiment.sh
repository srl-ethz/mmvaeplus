#!/bin/bash

OUTPUTDIR="../outputs"
EXPERIMENT="RobotActions_1"
DATADIR="../data"
EPOCHS=250
SEED=2
SHARED_LAT_DIM=32
MS_LAT_DIM=32

# Train MMVAEplus
python train_MMVAEplus_robot_actions.py --experiment $EXPERIMENT --obj "elbo" --K 1 --batch-size 128 --epochs $EPOCHS \
      --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta 2.5 \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --priorposterior "Normal"

