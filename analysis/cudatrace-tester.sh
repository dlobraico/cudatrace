#!/bin/bash

SEQ_BIN=c-ray-f
CUDA_BIN=cudatrace

SCENE_FILE="scene"

declare -a resolutions
resolutions=( 64x64 160x160 240x240 480x480 800x800 960x960 1120x1120 1280x1280 1440x1440 2400x2400 4800x4800 8000x8000 9600x9600 11200x11200 12800x12800 14400x14400 16000x16000 20000x20000 24000x24000 28000x28000 32000x32000 40000x40000 48000x48000 )

for res in ${resolutions[@]} 
do
    echo ${res}
    echo "================"
    VARS="COMPUTE_PROFILE=1 COMPUTE_PROFILE_CSV=1 COMPUTE_PROFILE_LOG=output/${res}-log.csv"
    BIN=${CUDA_BIN}
    #ARGS="-i ${SCENE_FILE} -o output/${BIN}-output-${res}.ppm -s ${res}"
    ARGS="-i ${SCENE_FILE} -s ${res}"

    export $VARS
    `./${BIN} ${ARGS} 1>/dev/null`

    BIN=${SEQ_BIN}
    #ARGS="-i ${SCENE_FILE} -o output/${BIN}-output-${res}.ppm -s ${res}"
    ARGS="-i ${SCENE_FILE} -s ${res}"

    `./${BIN} ${ARGS} 1>/dev/null`
    echo "================"
    echo
done
