#!/bin/bash
depth=5
epochs=10
height=128
width=128
batch=16

. ./parse_options.sh

name=Unet_${depth}_${epochs}

CUDA_VISIBLE_DEVICES=$(free-gpu) /home/yshao/miniconda2/bin/python \
		    ./DSB2018.py \
		    --name $name \
		    --depth $depth \
		    --batch-size $batch \
		    --img-height $height \
		    --img-width $width \
		    --epochs $epochs
