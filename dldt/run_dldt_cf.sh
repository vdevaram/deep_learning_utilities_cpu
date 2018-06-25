###########################################################################
# This is a sample script file to run TF inference benchmarks with 
# mkldnn library on Intel CPUs using DLDT tool
# Please contact vinod.devarampati@intel.com for any clarifications
# Instructions to fill the variable are given in comments
############################################################################
# Mandatory : Install latest DLDT from 
#        - https://software.intel.com/en-us/openvino-toolkit/choose-download
############################################################################
# Mention the linux Flavour
export LINUX="CENTOS"
#Point to the path where you want to load all files
export WKDIR=~/cf_inference_demo
#Point to the path where you want to load all images
export DATA_PATH=~/imageNet
export SAMPLES_PATH=$WKDIR/samples
export LOGS_PATH=$WKDIR/logs
export DLDT_PATH=~/intel/computer_vision_sdk_2018.1.265/deployment_tools
export MO_MODELS_PATH=$WKDIR/mo_models
export CAFFE_MODELS=$WKDIR/models
source  $DLDT_PATH/../bin/setupvars.sh
source $DLDT_PATH/model_optimizer/install_prerequisites/install_prerequisites_caffe.sh
source $DLDT_PATH/model_optimizer/install_prerequisites/../venv/bin/activate
mkdir -p $WKDIR
mkdir -p $MO_MODELS_PATH
mkdir -p $MO_MODELS_PATH
mkdir -p $MO_MODELS_PATH
mkdir -p $LOGS_PATH/BS1
mkdir -p $LOGS_PATH/BS16
mkdir -p $LOGS_PATH/BS32
mkdir -p $SAMPLES_PATH
cd $WKDIR
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
cd $WKDIR
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/bvlc_alexnet.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/alexnet_deploy.prototxt --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/bvlc_googlenet.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/googlenet_deploy.prototxt --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/VGG_ILSVRC_16_layers.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/VGG_ILSVRC_16_layers_deploy.prototxt --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/VGG_ILSVRC_19_layers.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/VGG_ILSVRC_19_layers_deploy.prototxt --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/ResNet-50-model.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/ResNet-50-deploy.prototxt  --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/ResNet-101-model.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/ResNet-101-deploy.prototxt  --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/ResNet-152-model.caffemodel --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/ResNet-152-deploy.prototxt  --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/inception-v3.caffemodel  --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/deploy_inception-v3.prototxt  --output_dir  $MO_MODELS_PATH --scale 255
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/inception-v4.caffemodel  --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/deploy_inception-v4.prototxt  --output_dir  $MO_MODELS_PATH --scale 255
python3 $DLDT_PATH/model_optimizer/mo.py --framework caffe --input_model $CAFFE_MODELS/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel  --batch 1 --data_type FP32  --input_proto $CAFFE_MODELS/ssd_vgg_deploy.prototxt  --output_dir  $MO_MODELS_PATH

cd $SAMPLES_PATH
cmake -DCMAKE_BUILD_TYPE=Release $DLDT_PATH/inference_engine/samples
make 
if [ -z $1 ]
  then
    NUMA=""
else
    NUMA="numactl -N $1 -m $1"
fi
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/VGG_ILSVRC_16_layers.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/$1_vgg_16.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/VGG_ILSVRC_19_layers.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/$1_vgg_19.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/bvlc_alexnet.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/$1_alexnet.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/bvlc_googlenet.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/$1_googlenet.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/ResNet-50-model.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/$1_resenet_v1_50.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/ResNet-101-model.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/$1_resenet_v1_101.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/ResNet-152-model.xml -d CPU -nt 2 -ni 1000  &>$LOGS_PATH/BS1/$1_resenet_v1_152.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/inception-v3.xml -d CPU -nt 2 -ni 1000 &>$LOGS_PATH/BS1/$1_inception_v3.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/1 -m $MO_MODELS_PATH/inception-v4.xml -d CPU -nt 2 -ni 1000 &>$LOGS_PATH/BS1/$1_inception_v4.log
$NUMA $SAMPLES_PATH/intel64/Release/object_detection_sample_ssd  -i $DATA_PATH/1  -m $MO_MODELS_PATH/VGG_VOC0712_SSD_300x300_iter_120000.xml  -d CPU -ni 1000 &>$LOGS_PATH/BS1/$1_ssd_vgg.log

$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/VGG_ILSVRC_16_layers.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/$1_vgg_16.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/VGG_ILSVRC_19_layers.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/$1_vgg_19.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/bvlc_alexnet.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/$1_alexnet.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/bvlc_googlenet.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/$1_googlenet.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/ResNet-50-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/$1_resenet_v1_50.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/ResNet-101-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/$1_resenet_v1_101.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/ResNet-152-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS16/$1_resenet_v1_152.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/inception-v3.xml -d CPU -nt 2 -ni 100 &>$LOGS_PATH/BS16/$1_inception_v3.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/16 -m $MO_MODELS_PATH/inception-v4.xml -d CPU -nt 2 -ni 100 &>$LOGS_PATH/BS16/$1_inception_v4.log

$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/VGG_ILSVRC_16_layers.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/$1_vgg_16.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/VGG_ILSVRC_19_layers.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/$1_vgg_19.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/bvlc_alexnet.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/$1_alexnet.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/bvlc_googlenet.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/$1_googlenet.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/ResNet-50-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/$1_resenet_v1_50.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/ResNet-101-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/$1_resenet_v1_101.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/ResNet-152-model.xml -d CPU -nt 2 -ni 100  &>$LOGS_PATH/BS32/$1_resenet_v1_152.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/inception-v3.xml -d CPU -nt 2 -ni 100 &>$LOGS_PATH/BS32/$1_inception_v3.log
$NUMA $SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH/32 -m $MO_MODELS_PATH/inception-v4.xml -d CPU -nt 2 -ni 100 &>$LOGS_PATH/BS32/$1_inception_v4.log
