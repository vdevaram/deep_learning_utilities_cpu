###########################################################################
# This is a sample script file to run TF inference benchmarks with 
# mkldnn library on Intel CPUs using DLDT tool
# Please contact vinod.devarampati@intel.com for any clarifications
# Instructions to fill the variable are given in comments
############################################################################
# Mandatory : Install latest DLDT and intel python 3.5 or later
############################################################################
#Point to the path where you want to load all files
export WKDIR=~/tf_inference_demo
#Point to the path where you want to load all images 
export DATA_PATH=~/imageNet
export SAMPLES_PATH=$WKDIR/samples
export LOGS_PATH=$WKDIR/logs
export DLDT_PATH=~/intel/deeplearning_deploymenttoolkit_2018.0.8585.0/deployment_tools
export PY3_PATH=/opt/intel/intelpython3
export InferenceEngine_DIR=$DLDT_PATH/inference_engine/share
export LD_LIBRARY_PATH=$PY3_PATH/lib:$DLDT_PATH/inference_engine/external/mklml_lnx/lib:$DLDT_PATH/inference_engine/external/cldnn/lib/:$DLDT_PATH/inference_engine/lib/centos_7.3/intel64/:$LD_LIBRARY_PATH
export PATH=$PY3_PATH/bin:$PATH
export MO_MODELS_PATH=$WKDIR/mo_models
export FROZEN_MODELS=$WKDIR/frozen
mkdir -p $WKDIR
mkdir -p $MO_MODELS_PATH
mkdir -p $SAMPLES_PATH
mkdir -p $LOGS_PATH
cd $WKDIR
echo "This script assumes that latest DLDT is installed. For Centos Intel python 3.5 or later is also installed"
#setup python environment
# For Ubuntu
# sudo apt install virtualenv
# virtualenv -p /usr/bin/python3 .dldt --system-site-packages 
sudo yum install python-virtualenv cmake
virtualenv -p /opt/intel/intelpython3/bin/python .dldt --system-site-packages
. .dldt/bin/activate
# download all trained .pb files
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
tar -xvf vgg_19_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvf inception_v3_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
tar -xvf inception_v4_2016_09_09.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xvf resnet_v1_50_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xvf resnet_v1_101_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
tar -xvf resnet_v1_152_2016_08_28.tar.gz
mkdir -p $WKDIR/ckpt
mv *.ckpt ckpt
rm -rf *.gz
#download the code
git clone https://github.com/tensorflow/models.git
#install tensorflow
pip3 install -r $DLDT_PATH/model_optimizer/requirements_tf.txt
mkdir -p $WKDIR/pb
mkdir -p $WKDIR/frozen
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=vgg_16 --output_file=$WKDIR/pb/vgg_16.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=vgg_19 --output_file=$WKDIR/pb/vgg_19.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=resnet_v1_50 --output_file=$WKDIR/pb/resnet_v1_50.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=resnet_v1_101 --output_file=$WKDIR/pb/resnet_v1_101.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=resnet_v1_152 --output_file=$WKDIR/pb/resnet_v1_152.pb  --labels_offset=1
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=inception_v3 --output_file=$WKDIR/pb/inception_v3.pb
python $WKDIR/models/research/slim/export_inference_graph.py  --alsologtostderr --model_name=inception_v4 --output_file=$WKDIR/pb/inception_v4.pb
cd $WKDIR/tensorflow
bazel build tensorflow/python/tools:freeze_graph
# summarize graph helps in giving details of node names for freeze.
#bazel build tensorflow/tools/graph_transforms:summarize_graph
#bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=$WKDIR/pb/resnet_v1_101.pb
# freeze graph to load into the inference program
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/vgg_16.pb   --input_checkpoint=$WKDIR/ckpt/vgg_16.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_vgg_16.pb   --output_node_names=vgg_16/fc8/squeezed
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/vgg_19.pb   --input_checkpoint=$WKDIR/ckpt/vgg_19.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_vgg_19.pb   --output_node_names=vgg_19/fc8/squeezed
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/resnet_v1_50.pb   --input_checkpoint=$WKDIR/ckpt/resnet_v1_50.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_resenet_v1_50.pb   --output_node_names=resnet_v1_50/predictions/Reshape_1
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/resnet_v1_101.pb   --input_checkpoint=$WKDIR/ckpt/resnet_v1_101.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_resenet_v1_101.pb   --output_node_names=resnet_v1_101/predictions/Reshape_1
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/resnet_v1_152.pb   --input_checkpoint=$WKDIR/ckpt/resnet_v1_152.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_resenet_v1_152.pb   --output_node_names=resnet_v1_152/predictions/Reshape_1
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/inception_v3.pb   --input_checkpoint=$WKDIR/ckpt/inception_v3.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_inception_v3.pb   --output_node_names=InceptionV3/Predictions/Reshape_1
bazel-bin/tensorflow/python/tools/freeze_graph   --input_graph=$WKDIR/pb/inception_v4.pb   --input_checkpoint=$WKDIR/ckpt/inception_v4.ckpt   --input_binary=true --output_graph=$WKDIR/frozen/frozen_inception_v4.pb   --output_node_names=InceptionV4/Logits/Predictions
cd $WKDIR
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model $FROZEN_MODELS/frozen_inception_v3.pb --batch 16 --data_type FP32 --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model $FROZEN_MODELS/frozen_inception_v4.pb --batch 16 --data_type FP32 --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model $FROZEN_MODELS/frozen_vgg_19.pb --batch 16 --data_type FP32 --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model $FROZEN_MODELS/frozen_vgg_16.pb --batch 16 --data_type FP32 --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model $FROZEN_MODELS/frozen_resenet_v1_50.pb --batch 16 --data_type FP32 --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model $FROZEN_MODELS/frozen_resenet_v1_101.pb --batch 16 --data_type FP32 --output_dir  $MO_MODELS_PATH
python3 $DLDT_PATH/model_optimizer/mo.py --framework tf --input_model $FROZEN_MODELS/frozen_resenet_v1_152.pb --batch 16 --data_type FP32 --output_dir  $MO_MODELS_PATH
cd $SAMPLES_PATH
cmake -DCMAKE_BUILD_TYPE=Release $DLDT_PATH/inference_engine/samples
make -j(nproc)
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH -m $MO_MODELS_PATH/frozen_vgg_16.xml -d CPU -nt 2  &>$LOGS_PATH/vgg_16.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH -m $MO_MODELS_PATH/frozen_vgg_19.xml -d CPU -nt 2  &>$LOGS_PATH/vgg_19.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH -m $MO_MODELS_PATH/frozen_inception_v3.xml -d CPU -nt 2  &>$LOGS_PATH/inception_v3.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH -m $MO_MODELS_PATH/frozen_inception_v4.xml -d CPU -nt 2  &>$LOGS_PATH/inception_v4.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH -m $MO_MODELS_PATH/frozen_resenet_v1_50.xml -d CPU -nt 2  &>$LOGS_PATH/resenet_v1_50.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH -m $MO_MODELS_PATH/frozen_resenet_v1_101.xml -d CPU -nt 2  &>$LOGS_PATH/resenet_v1_101.log
$SAMPLES_PATH/intel64/Release/classification_sample -i $DATA_PATH -m $MO_MODELS_PATH/frozen_resenet_v1_152.xml -d CPU -nt 2  &>$LOGS_PATH/resenet_v1_152.log
