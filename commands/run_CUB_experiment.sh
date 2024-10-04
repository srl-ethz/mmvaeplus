#!/bin/bash

OUTPUTDIR="../outputs"
EXPERIMENT="CUB_Image_Captions_1"
DATADIR="../data"
EPOCHS=50
SEED=2
SHARED_LAT_DIM=48
MS_LAT_DIM=16

# Train MMVAEplus
python train_MMVAEplus_CUB.py --experiment $EXPERIMENT --obj "dreg" --K 10 --batch-size 32 --epochs $EPOCHS \
      --latent-dim-z $SHARED_LAT_DIM --latent-dim-w $MS_LAT_DIM --seed $SEED --beta 1.0 \
      --datadir $DATADIR  --outputdir $OUTPUTDIR \
      --inception_path "${DATADIR}/pt_inception-2015-12-05-6726825d.pth" \
      --priorposterior "Normal"

