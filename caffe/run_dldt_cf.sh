###########################################################################
# This is a sample script file to run TF inference benchmarks with 
# mkldnn library on Intel CPUs using DLDT tool
# Please contact vinod.devarampati@intel.com for any clarifications
# Instructions to fill the variable are given in comments
############################################################################
# Mandatory : Install latest DLDT and intel python 3.5 or later
############################################################################
uname -a | grep -i 'Ubuntu' &>/dev/null
if [ $? -eq 0 ]; then export LINUX="UBUNTU"; else export LINUX="CENTOS"; fi
#Point to the path where you want to load all files
export WKDIR=~/cf_inference_demo
#Point to the path where you want to load all images 
export DATA_PATH=~/imageNet
export SAMPLES_PATH=$WKDIR/samples
export LOGS_PATH=$WKDIR/logs
export DLDT_PATH=~/intel/deeplearning_deploymenttoolkit_2018.0.8585.0/deployment_tools
export InferenceEngine_DIR=$DLDT_PATH/inference_engine/share
export MO_MODELS_PATH=$WKDIR/mo_models
export CAFFE_MODELS=$WKDIR/models
#setup python environment
if [ $LINUX == "UBUNTU" ]
then 
    export PY3_PATH=/usr/local 
else
    export PY3_PATH=/opt/intel/intelpython3
fi
export LD_LIBRARY_PATH=$PY3_PATH/lib:$DLDT_PATH/inference_engine/external/mklml_lnx/lib:$DLDT_PATH/inference_engine/external/cldnn/lib/:$DLDT_PATH/inference_engine/lib/centos_7.3/intel64/:$DLDT_PATH/inference_engine/lib/ubuntu_16.04/intel64/:$LD_LIBRARY_PATH
export PATH=$PY3_PATH/bin:$PATH
mkdir -p $WKDIR
mkdir -p $MO_MODELS_PATH
mkdir -p $MO_MODELS_PATH
mkdir -p $MO_MODELS_PATH
mkdir -p $LOGS_PATH/BS1
mkdir -p $LOGS_PATH/BS16
mkdir -p $LOGS_PATH/BS32
mkdir -p $SAMPLES_PATH
cd $WKDIR
echo "This script assumes that latest DLDT and OPENCV are installed. For Centos Intel python 3.5 or later is required installed"
if [ $LINUX == "UBUNTU" ]
then
    # For Ubuntu
    sudo apt update
    sudo apt install -y virtualenv cmake pkg-config zip g++ zlib1g-dev unzip 
    pip3 install setuptools numpy --upgrade 
    rm -rf .dldt
    virtualenv -p /usr/bin/python3 .dldt --system-site-packages 
    # Bazel for ubuntu
    bazel version &>/dev/null
    if ! [ $? -eq 0 ]
    then
        wget https://github.com/bazelbuild/bazel/releases/download/0.11.1/bazel-0.11.1-installer-linux-x86_64.sh
        chmod 777 bazel-0.11.1-installer-linux-x86_64.sh
        sudo ./bazel-0.11.1-installer-linux-x86_64.sh
    fi
else
    # For CentOS 
    bazel version &>/dev/null
    if ! [ $? -eq 0 ]
    then 
        wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo
	sudo mv vbatts-bazel-epel-7.repo /etc/yum.repos.d/
    fi
    sudo yum install -y python-virtualenv cmake numpy bazel
    sudo yum clean all
    virtualenv -p /opt/intel/intelpython3/bin/python .dldt --system-site-packages
fi
. .dldt/bin/activate
#download the models 
if ! [ -d "models" ]
then 
    mkdir -p $CAFFE_MODELS
    cd $CAFFE_MODELS
    wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
    wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
    wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
    wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt
    wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
    wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt  ; mv deploy.prototxt alexnet_deploy.prototxt
    wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
    wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt; mv deploy.prototxt googlenet_deploy.prototxt
    #download and keep the resnet models manually from the link : https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777
fi
pip install -r $DLDT_PATH/model_optimizer/requirements_caffe.txt
cd $WKDIR
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/bvlc_alexnet.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/alexnet_deploy.prototxt --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/bvlc_googlenet.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/googlenet_deploy.prototxt --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/VGG_ILSVRC_16_layers.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/VGG_ILSVRC_16_layers_deploy.prototxt --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/VGG_ILSVRC_19_layers.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/VGG_ILSVRC_19_layers_deploy.prototxt --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/ResNet-50-model.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/ResNet-50-deploy.prototxt  --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/ResNet-101-model.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/ResNet-101-deploy.prototxt  --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/ResNet-152-model.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/ResNet-152-deploy.prototxt  --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/inception-v3.caffemodel  --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/deploy_inception-v3.prototxt  --output_dir  $MO_MODELS_PATH --scale 255

cd $SAMPLES_PATH
cmake -DCMAKE_BUILD_TYPE=Release $DLDT_PATH/inference_engine/samples
make 
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/VGG_ILSVRC_16_layers.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/vgg_16.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/VGG_ILSVRC_19_layers.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/vgg_19.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/bvlc_alexnet.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/alexnet.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/bvlc_googlenet.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/googlenet.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/ResNet-50-model.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/resenet_v1_50.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/ResNet-101-model.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/resenet_v1_101.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/ResNet-152-model.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/resenet_v1_152.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/inception-v3.xml -d CPU -nt 2 -ni 1000 &>$LOGS_PATH/BS1/inception_v3.log

$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/VGG_ILSVRC_16_layers.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/vgg_16.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/VGG_ILSVRC_19_layers.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/vgg_19.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/bvlc_alexnet.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/alexnet.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/bvlc_googlenet.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/googlenet.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/ResNet-50-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/resenet_v1_50.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/ResNet-101-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/resenet_v1_101.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/ResNet-152-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/resenet_v1_152.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/inception-v3.xml -d CPU -nt 2 -ni 100 &>$LOGS_PATH/BS16/inception_v3.log

$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/VGG_ILSVRC_16_layers.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/vgg_16.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/VGG_ILSVRC_19_layers.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/vgg_19.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/bvlc_alexnet.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/alexnet.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/bvlc_googlenet.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/googlenet.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/ResNet-50-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/resenet_v1_50.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/ResNet-101-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/resenet_v1_101.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/ResNet-152-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/resenet_v1_152.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/inception-v3.xml -d CPU -nt 2 -ni 100 &>$LOGS_PATH/BS32/inception_v3.log
