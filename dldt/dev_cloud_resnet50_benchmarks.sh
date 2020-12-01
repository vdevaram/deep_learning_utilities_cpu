#Download Tensorflow benchmarks
git clone https://github.com/tensorflow/benchmarks.git

#setup opensource tensorflow
pip install virtualenv 
virtualenv -p python3 ~/.open_tf
source ~/.open_tf/bin/activate
pip install tensorflow
#get number of cores
export NUM_CORES=$(nproc)
# run training benchmarks
python ~/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 16 --mkl --num_inter_threads $NUM_CORES --num_inter_threads 1 --kmp_blocktime 1  --data_format NHWC

#### Queue submit and notedown FPS for training

# run inference benchmarks
python ~/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 16 --mkl --num_inter_threads $NUM_CORES --num_inter_threads 1 --kmp_blocktime 1  --data_format NHWC --forward_only

##### Queue submit and notedown FPS for inference

#deactivate opensource tensorflow environment
deactivate 

#setup intel tensorflow
source /opt/intel/inteloneapi/setvars.sh
source activate tensorflow
#get number of cores
export NUM_CORES=$(nproc)
# run training benchmarks
python ~/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 16 --mkl --num_inter_threads $NUM_CORES --num_inter_threads 1 --kmp_blocktime 1  --data_format NHWC

#### Queue submit and notedown FPS for training

# run inference benchmarks
python ~/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 16 --mkl --num_inter_threads $NUM_CORES --num_inter_threads 1 --kmp_blocktime 1  --data_format NHWC --forward_only

##### Queue submit and notedown FPS for inference

#deactivate intel  tensorflow environment
deactivate 


#Setup openvino environment
source  /opt/intel/openvino/bin/setupvars.sh
# create virtual environment and install openVINO dependencies 
virtualenv -p python3 ~/.openvino
source ~/.openvino/bin/activate
pip install networkx test-generator defusedxml
#build openvino samples 
mkdir ~/samples
cd ~/samples
cmake /opt/intel/openvino/deployment_tools/inference_engine/samples/cpp/
make -j

# get trained model from intel repo
cd 
wget https://download.01.org/opencv/public_models/012020/resnet-50-tf/resnet_v1-50.pb

#convert the tensorflow model to openVINO format
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_shape=[1,224,224,3] --mean_values=[123.68,116.78,103.94] --input=map/TensorArrayStack/TensorArrayGatherV3 --output=softmax_tensor --input_model=$HOME/resnet_v1-50.pb --reverse_input_channels

# run openVINO benchmarks
# run for batch size 1 and 1 stream and notedown FPS
$HOME/samples/intel64/Release/benchmark_app -m $HOME/resnet_v1-50.xml -nstreams 1 -niter 100 -b 1
# run for batch size 1 and 4 streams and notedown FPS
$HOME/samples/intel64/Release/benchmark_app -m $HOME/resnet_v1-50.xml -nstreams 4 -niter 100 -b 1
# run for batch size 16 and 1 stream and notedown FPS
$HOME/samples/intel64/Release/benchmark_app -m $HOME/resnet_v1-50.xml -nstreams 1 -niter 100 -b 16
# run for batch size 16 and 4 streams and notedown FPS
$HOME/samples/intel64/Release/benchmark_app -m $HOME/resnet_v1-50.xml -nstreams 4 -niter 100 -b 16

#deactivate openvino environment
deactivate 
