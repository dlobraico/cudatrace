#!/bin/bash

SEQ_BIN=c-ray-f
CUDA_BIN=cudatrace

SCENE_FILE="scene"

declare -a resolutions
declare -a threads
declare -a runs

resolutions=( 64x64 160x160 240x240 480x480 800x800 960x960 1120x1120 1280x1280 1440x1440 2400x2400 4800x4800 8000x8000 9600x9600 11200x11200 12800x12800 14400x14400 16000x16000 )
#resolutions=( 64x64 160x160 240x240 480x480 800x800 960x960 1120x1120 1280x1280 1440x1440 2400x2400 4800x4800 8000x8000 9600x9600 11200x11200 12800x12800 14400x14400 16000x16000 20000x20000 24000x24000 28000x28000 32000x32000 40000x40000 48000x48000 )
#resolutions=(24x24 64x64 128x128)
threads=( 1x1 2x2 4x4 8x8 12x12 16x16 20x20 22x22 )
runs=( 1 2 3 )

echo "xres,yres,xthreads,ythreads,render_time,real_time,application_type,run"

for run in ${runs[@]}
do
    for res in ${resolutions[@]} 
    do
        for thread_dims in ${threads[@]}
        do
            xres=$(echo ${res} | cut -d "x" -f "1")
            yres=$(echo ${res} | cut -d "x" -f "2")
            xthreads=$(echo ${thread_dims} | cut -d "x" -f "1")
            ythreads=$(echo ${thread_dims} | cut -d "x" -f "2")


            VARS="COMPUTE_PROFILE=1 COMPUTE_PROFILE_CSV=1 COMPUTE_PROFILE_LOG=output/${res}-${thread_dims}-${run}-log.csv"
            BIN=${CUDA_BIN}
            #ARGS="-i ${SCENE_FILE} -o output/${BIN}-output-${res}-${thread_dims}-run${run}.ppm -s ${res}"
            ARGS="-i ${SCENE_FILE} -s ${res} -t ${thread_dims}"

            export $VARS
            t1=$(($(date +%s%N)/1000000))
            render_time=$(./${BIN} ${ARGS} 2>&1 >&- | cut -d " " -f "5" | cut -d "(" -f "2")
            t2=$(($(date +%s%N)/1000000))
            real_time=$((${t2}-${t1}))
            echo "${xres},${yres},${xthreads},${ythreads},${render_time},${real_time},cuda,${run}"

        done

        BIN=${SEQ_BIN}
        #ARGS="-i ${SCENE_FILE} -o output/${BIN}-output-${res}-${thread_dims}-run${run}.ppm -s ${res}"
        ARGS="-i ${SCENE_FILE} -s ${res}"

        t1=$(($(date +%s%N)/1000000))
        render_time=$(./${BIN} ${ARGS} 2>&1 >&- | cut -d " " -f "5" | cut -d "(" -f "2")
        t2=$(($(date +%s%N)/1000000))
        real_time=$((${t2}-${t1}))
        echo "${xres},${yres},,,${render_time},${real_time},sequential,${run}"
    done
done
