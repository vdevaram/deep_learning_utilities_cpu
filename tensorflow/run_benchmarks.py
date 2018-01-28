import os
import socket

#ps_nodes = [ "10.250.0.45"]#, "10.250.0.47"]#, "10.250.0.46", "10.250.0.47" ]
#worker_nodes = [ "10.250.0.43", "10.250.0.44", "10.250.0.46", "10.250.0.47", "10.250.0.35", "10.250.0.39", "10.250.0.40", "10.250.0.41" ]
#ps_names = ["wk1", "wk2", "wk3","wk4"]#"wk4", "wk8", "wk12", "wk16"]
ps_names = ["ps"]
worker_names = ["ps"]#"wk2"]#, "wk3","wk4"]#, "wk2", "wk3", "wk4"]
#worker_names = ["wk1", "wk2", "wk3", "wk4"]
ps_nodes = [socket.gethostbyname(name) for name in ps_names ]
worker_nodes = [ socket.gethostbyname(name) for name in worker_names]
num_workers_per_node = 12
num_ps_per_node = 1
models = { "alexnet":"512", "googlenet":"64", "resnet50":"64", "inception3":"64", "resnet152":"64", "vgg16":"64" }
#models = { "resnet50":"64"} #, "resnet50":"64", "resnet152":"64", "vgg16":"64" }
envfile_path = "~/work/tf/tf.sh"
ps_numa_cmd = { 1:["numactl -l"], 2:["numactl -N 0 -m 0", "numactl -N 1 -m 1"] }
wk_numa_cmd = {1:["numactl -l"], \
            2:["numactl -N 0 -m 0", "numactl -N 1 -m 1"], \
            4:['export KMP_AFFINITY="granularity=thread,proclist=[0-11,48-59],explicit,verbose";numactl -m 0 ', \
               'export KMP_AFFINITY="granularity=thread,proclist=[12-23,60-71],explicit,verbose"; numactl -m 0 ', \
               'export KMP_AFFINITY="granularity=thread,proclist=[24-35,72-83],explicit,verbose"; numactl -m 1 ', \
               'export KMP_AFFINITY="granularity=thread,proclist=[36-47,85-95],explicit,verbose"; numactl -m 1'],\
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
common_envs = [ "export TF_ADJUST_HUE_FUSED=1", "export TF_ADJUST_SATURATION_FUSED=1"]
#ps_envs = ["export OMP_NUM_THREADS=4"] + common_envs
ps_envs = common_envs
#ps_envs = ["export OMP_NUM_THREADS=4", 'export KMP_AFFINITY="granularity=thread,proclist=[92-95],explicit,verbose"'] + common_envs
#worker_envs = ["export OMP_NUM_THREADS=22",  'export KMP_AFFINITY="granularity=thread,proclist=[0-21],explicit,verbose"'] + common_envs
#worker_envs =  common_envs
worker_envs = ["export OMP_NUM_THREADS=8"] + common_envs
script_path = "~/work/tf/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py"
data_format = "NCHW"
num_batches = "40"
other_params = "--distortions False --local_parameter_device cpu --device cpu --mkl True kmp_blocktime 1"
ps_threads = ["4","2"]
worker_threads = ["8", "2"]
data_params = ["~/work/tf/tf_records", "imagenet"]
protocol = "grpc"
log_path = "~/work/tf"
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
    print "sleep 15m"

