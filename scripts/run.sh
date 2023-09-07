#!/bin/bash

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash run.sh NETWORK MINDIR_PATH DATASET DATA_PATH [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

network=$1
mindir_path=$(get_real_path "$2")
dataset=$3
data_path=$(get_real_path "$4")
device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi
output_path="outputs"
echo "network: $network"
echo "mindir_path: $mindir_path"
echo "dataset: $dataset"
echo "data_path: $data_path"
echo "device_id: $device_id"
echo "output_path: $output_path"

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin:${ASCEND_HOME}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/ascend-toolkit/latest/opp
    export ASCEND_AICPU_PATH=${ASCEND_HOME}/ascend-toolkit/latest/
    export PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/python/site-packages:${PYTHONPATH}
    export TOOLCHAIN_HOME=${ASCEND_HOME}/ascend-toolkit/latest/toolkit
else
    export PATH=${ASCEND_HOME}/latest/compiler/bin:${ASCEND_HOME}/latest/compiler/ccec_compiler/bin:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/latest/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
    export ASCEND_AICPU_PATH=${ASCEND_HOME}/latest
    export PYTHONPATH=${ASCEND_HOME}/latest/compiler/python/site-packages:${PYTHONPATH}
    export TOOLCHAIN_HOME=${ASCEND_HOME}/latest/toolkit
fi

# export MS_LITE_HOME=/home/stu/wyf/mindspore-lite-2.0.0-linux-x64
# if [ $MS_LITE_HOME ]; then
#     RUNTIME_HOME=$MS_LITE_HOME/runtime
#     TOOLS_HOME=$MS_LITE_HOME/tools
#     RUNTIME_LIBS=$RUNTIME_HOME/lib:$RUNTIME_HOME/third_party/glog/:$RUNTIME_HOME/third_party/libjpeg-turbo/lib
#     RUNTIME_LIBS=$RUNTIME_LIBS:$RUNTIME_HOME/third_party/dnnl/
#     export LD_LIBRARY_PATH=$RUNTIME_LIBS:$TOOLS_HOME/converter/lib:$LD_LIBRARY_PATH
#     echo "Insert LD_LIBRARY_PATH the MindSpore Lite runtime libs path: $RUNTIME_LIBS $TOOLS_HOME/converter/lib"
# fi

# function convert_model()
# {
#     $MS_LITE_HOME/tools/converter/converter/converter_lite --fmk=MINDIR --modelFile=$mindir_path --outputFile=$mindir_path.ms
# }

function compile_app()
{
    if [ $MS_LITE_HOME ];then
        MINDSPORE_PATH=$MS_LITE_HOME/runtime
    else
        MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
        if [[ ! $MINDSPORE_PATH ]];then
            MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
        fi
    fi
    build_path="build"
    if [ -d $build_path ]; then
        rm -rf $build_path
    fi
    mkdir -p $build_path && cd $build_path
    cmake .. -DMINDSPORE_PATH=$MINDSPORE_PATH
    make -j8
    cd ..
}

function infer()
{
    if [ -d $output_path ]; then
        rm -rf $output_path
    fi
    mkdir -p $output_path
    ./build/main \
        --network=$network --mindir_path=$mindir_path \
        --dataset=$dataset --data_path=$data_path --output_path=$output_path \
        --device_type=Ascend --device_id=$device_id  &> infer.log
}

function cal_acc()
{
    python ./scripts/postprocess.py --dataset_path=$data_path --results_path=$output_path &> acc.log
}

# echo -e "\e[1;36mConverting...\e[0m"
# convert_model
# if [ $? -ne 0 ]; then
#     echo "convert model failed"
#     exit 1
# fi
echo -e "\e[1;36mCompiling...\e[0m"
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
echo -e "\e[1;36mInfering...\e[0m"
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
echo -e "\e[1;36mCalculating...\e[0m"
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
