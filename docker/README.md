# Instructions for using the docker-ce for Deep learning on Intel CPUs
* official guide for instalaltion of [dockers](https://docs.docker.com/engine/installation/linux/docker-ce/centos/)
* download and source the file **install_docker_centos.sh** to install docker on the centos
* Create folder *workdir* and change dir **cd workdir**
* Create file “Dockerfile” and write all the docker commands in it. 
* keep all the required libs and scripts mentioned in Dockerfile
* Use the below command to generate docker image 
   ```python
   docker build  -t <image_name> .
   ```
* To remove intermediate images after getting successful image
  ```python
  docker rmi $(docker images -f "dangling=true" -q)
  ```
* Container launch
  ```python
  docker run -t --name <container name> <image name>  
  ```
* Container launch with *interactive mode* 
  ```python
  docker run -it --name <container name> <image name>
  ```
* Container launch for *direct execution* 
  ```python
  docker run -t --name <container name> <image name> /bin/bash -c "<shell commands to be executed>"
  ```
* Container launch with *auto delete* after execution 
  ```python
  docker run -t --rm --name <container name> <image name> /bin/bash -c "<shell commands to be executed>"
  ```
* Container will have root permissions and anything can be installed by user
* Data changed and new installations are ephemeral. So we need to attach a persistent storage to retain data. There are three ways to do it.
  * **VOLUME** command in Dockerfile
  * **--mount** option to attach a docker volume 
  * **-v** option to use local disk path 
* Docker command for attaching local path
  ```python
  docker run -it --name <Container name> -v <src path>:<dst path> <image name> 
  ```
* By deafult container will have limited previlages to use Networking and linux capabilities. So use **previleged mode** for full control like bare metal
  ```python
  docker run -it --name <Container name> --network=host  --privileged -v <src path>:<dst path> <image name>
  ```
