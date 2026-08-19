#!/bin/bash

echo "=== Iniciando Sistema Escravo (Ubuntu) ==="

# 1. Inicia os sensores físicos (Câmera e Garra)
echo ">> Iniciando câmera RealSense D435 e Garra Robotiq..."
roslaunch realsense2_camera rs_camera.launch align_depth.enable:=true &
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0 &
sleep 5

# 2. Inicia o driver do UR5, o MoveIt e a pipeline de visão (VoxelGrid/PassThrough)
echo ">> Iniciando controle do manipulador e lógica de autonomia compartilhada..."
# Substitua 'nome_do_seu_arquivo.launch' pelo nome exato que você deu ao arquivo .launch
roslaunch UR5_VR_pkg real_ur5_gripper_140_cam_noRViz.launch &
sleep 5

# 3. Habilita a ponte de comunicação apontando para o adaptador USB (Roteador)
echo ">> Estabelecendo conexão ROS-TCP para o Unity (IP: 192.168.0.100)..."
roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=192.168.0.100 tcp_port:=10000 &

echo "=== Sistema pronto para receber comandos do Unity ==="
wait
