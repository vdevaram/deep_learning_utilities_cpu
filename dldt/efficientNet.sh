# This file converts efficientNet models to OpenVINO
# set the variables for openVINO
export OPENVINO_INSTALL_DIR=$HOME
export SAMPLES_DIR=$HOME/samples

# set workspace
export WORKSPACE=$HOME/efficientNet
mkdir $HOME/efficientNet
cd $HOME/efficientNet
source $OPENVINO_INSTALL_DIR/intel/openvino/bin/setupvars.sh

#set CPU settings
export CORES_PER_SOCKET=24


wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b1.tar.gz
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b2.tar.gz
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b3.tar.gz
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b4.tar.gz
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b5.tar.gz
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b6.tar.gz
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b7.tar.gz

tar -zxvf efficientnet-b0.tar.gz
tar -zxvf efficientnet-b1.tar.gz 
tar -zxvf efficientnet-b2.tar.gz 
tar -zxvf efficientnet-b3.tar.gz 
tar -zxvf efficientnet-b4.tar.gz 
tar -zxvf efficientnet-b5.tar.gz 
tar -zxvf efficientnet-b6.tar.gz 
tar -zxvf efficientnet-b7.tar.gz 

freeze_graph  --output_node_names=efficientnet-b0/model/head/dense/BiasAdd --output_graph=efficientnet-b0/efficientnet_b0.pb --input_meta_graph=efficientnet-b0/model.ckpt.meta --input_binary=true --input_checkpoint=efficientnet-b0/model.ckpt
freeze_graph  --output_node_names=efficientnet-b1/model/head/dense/BiasAdd --output_graph=efficientnet-b1/efficientnet_b1.pb --input_meta_graph=efficientnet-b1/model.ckpt.meta --input_binary=true --input_checkpoint=efficientnet-b1/model.ckpt
freeze_graph  --output_node_names=efficientnet-b2/model/head/dense/BiasAdd --output_graph=efficientnet-b2/efficientnet_b2.pb --input_meta_graph=efficientnet-b2/model.ckpt.meta --input_binary=true --input_checkpoint=efficientnet-b2/model.ckpt
freeze_graph  --output_node_names=efficientnet-b3/model/head/dense/BiasAdd --output_graph=efficientnet-b3/efficientnet_b3.pb --input_meta_graph=efficientnet-b3/model.ckpt.meta --input_binary=true --input_checkpoint=efficientnet-b3/model.ckpt
freeze_graph  --output_node_names=efficientnet-b4/model/head/dense/BiasAdd --output_graph=efficientnet-b4/efficientnet_b4.pb --input_meta_graph=efficientnet-b4/model.ckpt.meta --input_binary=true --input_checkpoint=efficientnet-b4/model.ckpt
freeze_graph  --output_node_names=efficientnet-b5/model/head/dense/BiasAdd --output_graph=efficientnet-b5/efficientnet_b5.pb --input_meta_graph=efficientnet-b5/model.ckpt.meta --input_binary=true --input_checkpoint=efficientnet-b5/model.ckpt
freeze_graph  --output_node_names=efficientnet-b6/model/head/dense/BiasAdd --output_graph=efficientnet-b6/efficientnet_b6.pb --input_meta_graph=efficientnet-b6/model.ckpt.meta --input_binary=true --input_checkpoint=efficientnet-b6/model.ckpt
freeze_graph  --output_node_names=efficientnet-b7/model/head/dense/BiasAdd --output_graph=efficientnet-b7/efficientnet_b7.pb --input_meta_graph=efficientnet-b7/model.ckpt.meta --input_binary=true --input_checkpoint=efficientnet-b7/model.ckpt

python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b0/efficientnet_b0.pb  --input "IteratorGetNext:0[1 300 300 3]" --output_dir ./300_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b1/efficientnet_b1.pb  --input "IteratorGetNext:0[1 300 300 3]" --output_dir ./300_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b2/efficientnet_b2.pb  --input "IteratorGetNext:0[1 300 300 3]" --output_dir ./300_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b3/efficientnet_b3.pb  --input "IteratorGetNext:0[1 300 300 3]" --output_dir ./300_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b4/efficientnet_b4.pb  --input "IteratorGetNext:0[1 300 300 3]" --output_dir ./300_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b5/efficientnet_b5.pb  --input "IteratorGetNext:0[1 300 300 3]" --output_dir ./300_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b6/efficientnet_b6.pb  --input "IteratorGetNext:0[1 300 300 3]" --output_dir ./300_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b7/efficientnet_b7.pb  --input "IteratorGetNext:0[1 300 300 3]" --output_dir ./300_xml

python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b0/efficientnet_b0.pb  --input "IteratorGetNext:0[1 600 600 3]" --output_dir ./600_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b1/efficientnet_b1.pb  --input "IteratorGetNext:0[1 600 600 3]" --output_dir ./600_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b2/efficientnet_b2.pb  --input "IteratorGetNext:0[1 600 600 3]" --output_dir ./600_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b3/efficientnet_b3.pb  --input "IteratorGetNext:0[1 600 600 3]" --output_dir ./600_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b4/efficientnet_b4.pb  --input "IteratorGetNext:0[1 600 600 3]" --output_dir ./600_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b5/efficientnet_b5.pb  --input "IteratorGetNext:0[1 600 600 3]" --output_dir ./600_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b6/efficientnet_b6.pb  --input "IteratorGetNext:0[1 600 600 3]" --output_dir ./600_xml
python3 $OPENVINO_INSTALL_DIR/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model efficientnet-b7/efficientnet_b7.pb  --input "IteratorGetNext:0[1 600 600 3]" --output_dir ./600_xml

numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./300_xml/efficientnet_b0.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b0.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./300_xml/efficientnet_b1.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b1.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./300_xml/efficientnet_b2.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b2.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./300_xml/efficientnet_b3.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b3.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./300_xml/efficientnet_b4.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b4.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./300_xml/efficientnet_b5.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b5.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./300_xml/efficientnet_b6.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b6.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./300_xml/efficientnet_b7.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b7.log

numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./600_xml/efficientnet_b0.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b0.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./600_xml/efficientnet_b1.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b1.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./600_xml/efficientnet_b2.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b2.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./600_xml/efficientnet_b3.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b3.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./600_xml/efficientnet_b4.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b4.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./600_xml/efficientnet_b5.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b5.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./600_xml/efficientnet_b6.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b6.log
numactl -m 0 -N 0 -C 0-$CORES_PER_SOCKET-1  $SAMPLES_DIR/intel64/Release/benchmark_app -m ./600_xml/efficientnet_b7.xml --nstreams 1 --niter 100 &>./logs/efficientnet_b7.log
