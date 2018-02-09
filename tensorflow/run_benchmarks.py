###########################################################################
# This is a sample script file to run TF benchmarks with mkldnn library on Intel CPUs
# Please contact vinod.devarampati@intel.com for any clarifications
# Instructions to fill the variable are given in comments
# make sure you are using TF1.5 or later to use these scripts
# Sample files required in this script are given below :
############################################################################
# sample kp.sh
# 	ps -elf | grep  tf_cnn | for i in $(awk '{print $4}');do kill $i; done
############################################################################
# Sample tf.sh
# 	export GCC_PATH=~/work/tf/gcc
#	export MKL_PATH=~/work/tf/mkl
#	export PATH=$GCC_PATH/bin:$PATH
#	export LD_LIBRARY_PATH=$MKL_PATH/lib:$GCC_PATH/lib64:$LD_LIBRARY_PATH
#	export KMP_BLOCKTIME=1;
#############################################################################

import os
import socket

# Enter the Hostname of each parameter server as a list
# Example : ps_names = ["ps1", "ps2", "ps3","ps4"]
ps_names = ["skl1","skl2","skl3","skl4"]
# Enter the Hostname of all workers as a list
#Example : worker_names = ["wk1", "wk2", "wk3", "wk4"]
worker_names = ["skl1","skl2","skl3","skl4"]
# Fill the below variables if you wish to fill only IP addresses rather than hostnames.
if ps_names == []:
    # Enter the Ipa address  of all PS servers as a list
    # Example: ps_nodes = [ "10.250.0.45", "10.250.0.46" ]
    ps_nodes = []
else :
    ps_nodes = [socket.gethostbyname(name) for name in ps_names ]
if worker_names == [] :
    # Enter the IP address of all workers as a list
    # Example : worker_nodes = [ "10.250.0.43", "10.250.0.44"]
    worker_nodes = []
else:
    worker_nodes = [ socket.gethostbyname(name) for name in worker_names]

if ps_nodes == [] or worker_nodes == [] :
    print(" please enter the hostname or IP address of ps and worker servers")

# Change num_workers_per_node if multiple workers are ran per node. 
# Please check numa and cpu commands and edit accordingly for multi worker per node
num_workers_per_node = 1
# Change  num_ps_per_node if multiple ps per node
num_ps_per_node = 1
# Add all the neural net topologies with their batch sizes to models dictionary
# Example : models = { "alexnet":"512", "googlenet":"64", "resnet50":"64", "inception3":"64", "resnet152":"64", "vgg16":"64" }
#models = { "alexnet":"512", "googlenet":"64", "resnet50":"64", "inception3":"64", "resnet152":"64", "vgg16":"64" }
models = { "vgg16":"64" }
# Keep all the environmental settings for benchamrking in a file and set the path to envfile_path
# Example : envfile_path = "~/work/tf/tf.sh"
envfile_path = "~/work/tf/tf.sh"
# Enter the NUMA commands and cpu affinity commands for PS and workers as shown below.
# --> Key of the ps_numa_cmd represents number of PS per node.
# --> Value represents command for each worker. Enter the whole command under quotes, separate each command with semicolon
# Example : ps_numa_cmd = { 1:["numactl -l"], 2:["numactl -N 0 -m 0", "numactl -N 1 -m 1"] }
# NOTE: You need to modify the proclist in the below commands depending on the Xeon server used.
# Example command is given below for Xeon SP 8168
ps_numa_cmd = { 1:["numactl -l"],
                2:["numactl -N 0 -m 0", "numactl -N 1 -m 1"] }

wk_numa_cmd = { 1:["numactl -l"], \
        2:["numactl -N 0 -m 0", "numactl -N 1 -m 1"], \
        4:['export KMP_AFFINITY="granularity=thread,proclist=[0-9,40-49],explicit,verbose";numactl -m 0 ', \
           'export KMP_AFFINITY="granularity=thread,proclist=[10-19,50-59],explicit,verbose"; numactl -m 0 ', \
           'export KMP_AFFINITY="granularity=thread,proclist=[20-29,60-69],explicit,verbose"; numactl -m 1 ', \
           'export KMP_AFFINITY="granularity=thread,proclist=[30-39,70-79],explicit,verbose"; numactl -m 1'],\
        8:['export KMP_AFFINITY="granularity=thread,proclist=[0-11,96-107],explicit,verbose";  numactl -m 0 ', \
           'export KMP_AFFINITY="granularity=thread,proclist=[12-23,108-119],explicit,verbose"; numactl -m 0 ', \
           'export KMP_AFFINITY="granularity=thread,proclist=[24-35,120-131],explicit,verbose"; numactl -m 1 ', \
           'export KMP_AFFINITY="granularity=thread,proclist=[36-47,132-143],explicit,verbose"; numactl -m 1 ',\
           'export KMP_AFFINITY="granularity=thread,proclist=[48-59,144-155],explicit,verbose"; numactl -m 2 ',\
           'export KMP_AFFINITY="granularity=thread,proclist=[60-71,156-167],explicit,verbose"; numactl -m 2 ',\
           'export KMP_AFFINITY="granularity=thread,proclist=[72-83,168-179],explicit,verbose"; numactl -m 3 ',\
           'export KMP_AFFINITY="granularity=thread,proclist=[84-95,180-191],explicit,verbose"; numactl -m 3'],\
        12:[' export KMP_AFFINITY="granularity=thread,proclist=[0-7,96-103],explicit,verbose"; numactl -m 0 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[8-15,104-111],explicit,verbose"; numactl -m 0 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[16-23,112-119],explicit,verbose"; numactl -m 0 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[24-31,120-127],explicit,verbose"; numactl -m 1 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[32-39,128-135],explicit,verbose"; numactl -m 1 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[40-47,136-143],explicit,verbose"; numactl -m 1 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[48-55,144-151],explicit,verbose"; numactl -m 2 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[56-63,152-159],explicit,verbose"; numactl -m 2 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[64-71,160-167],explicit,verbose"; numactl -m 2 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[72-79,168-175],explicit,verbose"; numactl -m 3 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[80-87,176-183],explicit,verbose"; numactl -m 3 ', \
            ' export KMP_AFFINITY="granularity=thread,proclist=[88-95,184-191],explicit,verbose"; numactl -m 3 '] }

# Add any common envs relavant for both PS and workers
common_envs = [ "export TF_ADJUST_HUE_FUSED=1", "export TF_ADJUST_SATURATION_FUSED=1"]
# Add any PS specific common envs
ps_envs = [] + common_envs
# PS intra and inter threads
ps_threads = ["4","2"]
# Add any worker specific common envs
worker_envs = ["export OMP_NUM_THREADS=10"] + common_envs
# worker intra and inter threads
worker_threads = ["9", "2"]
# Enter the path of the benchamrks script
script_path = "~/work/tf/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py"
# Add all other options required by the scripts
data_format = "NCHW"
num_batches = "40"
other_params = "--distortions=False --local_parameter_device cpu --device cpu --mkl=True "
data_params = ["~/work/tf/tf_records", "imagenet"]
protocol = "grpc"
# Path of the logfile to store output logs
log_path = "~/"

ps_hosts = ""
port = 2220

for node in ps_nodes:
    for i in range(num_ps_per_node):
        ps_hosts = ps_hosts + node + ":" + str(port)
        if not (node == ps_nodes[-1] and i == num_ps_per_node-1):
            ps_hosts = ps_hosts+ ","
        port = port + 1

wk_hosts = ""
port = 4440
for node in worker_nodes:
    for i in range(num_workers_per_node):
        wk_hosts = wk_hosts + node + ":" + str(port)
        if not (node == worker_nodes[-1] and i == num_workers_per_node-1):
            wk_hosts = wk_hosts + ","
        port = port + 1

for model in models:
    idx = 0
    for node in ps_nodes:
        print 'ssh',node,"'./kp.sh;'\n"
    for node in worker_nodes:
        print 'ssh',node,"'./kp.sh;'\n"
    for node in ps_nodes:
        for numa in ps_numa_cmd[num_ps_per_node]:
            cmd = "source " + envfile_path + "; "
            for var in ps_envs:
                cmd = cmd + var + "; "
            path = os.path.join(log_path , model + "ps" + str(idx) + ".log")
            cmd = cmd + numa + " "
            cmd = cmd + " python -u " + script_path + " "
            cmd = cmd + "--model " + model + " "
            cmd = cmd + "--batch_size " + models[model] + " "
            cmd = cmd + "--data_format " + data_format + " "
            cmd = cmd + "--num_batches " + num_batches + " "
            cmd = cmd + other_params + " "
            cmd = cmd + "--num_intra_threads " + ps_threads[0] + " "
            cmd = cmd + "--num_inter_threads " + ps_threads[1] + " "
            cmd = cmd + "--data_dir " + data_params[0] + " "
            cmd = cmd + "--data_name " + data_params[1] + " "
            cmd = cmd + "--server_protocol " + protocol + " "
            cmd = cmd + "--ps_hosts " + ps_hosts + " "
            cmd = cmd + "--worker_hosts " + wk_hosts + " "
            cmd = cmd + "--job_name ps" + " "
            cmd = cmd + "--task_index " + str(idx) + " "
            cmd = cmd + "&>"+ path + " "
            idx = idx + 1
            print "ssh ", node , " '", cmd ,"& '\n"

    idx = 0
    for node in worker_nodes:
        for numa in wk_numa_cmd[num_workers_per_node]:
            cmd = "source " + envfile_path + "; "
            for var in worker_envs:
                cmd = cmd + var + "; "
            path = os.path.join(log_path , model + "wk" + str(idx) + ".log")
            cmd = cmd + numa + " "
            cmd = cmd + " python -u " + script_path + " "
            cmd = cmd + "--model " + model + " "
            cmd = cmd + "--batch_size " + models[model] + " "
            cmd = cmd + "--data_format " + data_format + " "
            cmd = cmd + "--num_batches " + num_batches + " "
            cmd = cmd + other_params + " "
            cmd = cmd + "--num_intra_threads " + worker_threads[0] + " "
            cmd = cmd + "--num_inter_threads " + worker_threads[1] + " "
            cmd = cmd + "--data_dir " + data_params[0] + " "
            cmd = cmd + "--data_name " + data_params[1] + " "
            cmd = cmd + "--server_protocol " + protocol + " "
            cmd = cmd + "--ps_hosts " + ps_hosts + " "
            cmd = cmd + "--worker_hosts " + wk_hosts + " "
            cmd = cmd + "--job_name worker" + " "
            cmd = cmd + "--task_index " + str(idx) + " "
            cmd = cmd + "&>" + path + " "
            idx = idx + 1
            print "ssh ", node , " '", cmd ,"& '\n"
    print "echo 'waiting for ",model," to finish'"
    print "sleep 10m"
