# This file consists of instructions to download and run BERT demo with OpenVINO container on Intel CPUs
# download intel OpenVINO container
docker pull openvino/ubuntu18_dev:2021.1
# consider workspace to be bert_demo
export workspace=$HOME/bert_demo
mkdir $workspace

# download OpenVINO bert IR files (.xml and .bin ) from open_model_zoo 
cd  $workspace
# download FP32 files
wget https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/bert-large-uncased-whole-word-masking-squad-fp32-0001/FP32/bert-large-uncased-whole-word-masking-squad-fp32-0001.xml
wget https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/bert-large-uncased-whole-word-masking-squad-fp32-0001/FP32/bert-large-uncased-whole-word-masking-squad-fp32-0001.bin
# Download INT8 files 
wget https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/bert-large-uncased-whole-word-masking-squad-int8-0001/FP32-INT8/bert-large-uncased-whole-word-masking-squad-int8-0001.xml
wget https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/bert-large-uncased-whole-word-masking-squad-int8-0001/FP32-INT8/bert-large-uncased-whole-word-masking-squad-int8-0001.bin
# download vocab file for words interpretation 
wget https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/intel/bert-large-uncased-whole-word-masking-squad-fp32-0001/vocab.txt
# run a OpenVINO container to benchmark and demo 
docker run -it -v $workspace:/workspace  openvino/ubuntu18_dev:2021.1
cd /workspace/
# needed benchmark_app for benchmarking which needs to built from samples folder in OpenVINO install /opt/intel
mkdir samples
cd samples
cmake /opt/intel/openvino/deployment_tools/inference_engine/samples/cpp
make -j
# benchmark_app is available in /workspace/samples/intel64/Release/. Run as below to get help 
 /workspace/samples/intel64/Release/benchmark_app -h
# sample execution of benchmark app to know FPS and latency of downloaded FP32 model 
 /workspace/samples/intel64/Release/benchmark_app -m /workspace/bert-large-uncased-whole-word-masking-squad-fp32-0001.xml -niter 100 -nstreams 1
# sample execution of benchmark app to know FPS and latency of downloaded INT8 model 
 /workspace/samples/intel64/Release/benchmark_app -m /workspace/bert-large-uncased-whole-word-masking-squad-int8-0001.xml -niter 100 -nstreams 1
# BERT demo is available in python_demos folder of openvino install path. use help to know options
python3 /opt/intel/openvino/deployment_tools/inference_engine/demos/python_demos/bert_question_answering_demo/bert_question_answering_demo.py -h
# Run demo with proper options. A successful run will give prompt to enter qustions and after entering it will search whole text in given link and gives the answer with confidence metrics
python3 /opt/intel/openvino/deployment_tools/inference_engine/demos/python_demos/bert_question_answering_demo/bert_question_answering_demo.py -v /workspace/vocab.txt -m /workspace/bert-large-uncased-whole-word-masking-squad-int8-0001.xml -i https://en.wikipedia.org/wiki/Intel --input_names "result.1,result.2,result.3"  --output_names 5211,5212
# end 
