#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray # Para receber o Goal simples do Unity


# NOVO: Importa a classe de cálculo de trajetória (Assumindo arquivo trajectory_generator.py)
from trajectory_generator import generate_smooth_trajectory 

# --- CONFIGURAÇÕES DE JUNTAS ---
JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
GRIPPER_JOINT_NAME = 'finger_joint'
ALL_JOINT_NAMES = JOINT_NAMES + [GRIPPER_JOINT_NAME] # [7 Juntas]

# --- PARÂMETROS DE RESET SUAVE ---
RESET_TIME_SECONDS = 4.0  # Tempo total para o Reset (DEVE CORRESPONDER AO Unity: 4.0s - 0.5s buffer)
PUB_FREQUENCY_HZ = 50.0   # Frequência de pontos da trajetória
RESET_GOAL_TOPIC = 'reset_pose_request' 

# --- VARIÁVEIS DE ESTADO ---
latest_joint_state = None # Armazena a última pose de teleoperação conhecida (JointState)

rospy.init_node('unity_joint_publisher_node')

# Publisher para o controlador de trajetória do UR5 e da garra
pub_ur5 = rospy.Publisher('/ur5/eff_joint_traj_controller/command', JointTrajectory, queue_size=10)
pub_gripper = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=10)
# NOVO: Publisher para enviar os pontos da trajetória para o Unity
pub_unity_traj = rospy.Publisher('/reset_trajectory_points', JointTrajectory, queue_size=1)


# --- CALLBACKS ---

# No script unity_joint_publisher.py (Ubuntu)

def unity_joint_state_callback(data):
    """
    Callback de Teleoperação: Recebe JointState do Unity e retransmite (rápido).
    """
    global latest_joint_state # <-- ADICIONE OU GARANTA QUE ESTA LINHA EXISTA
    
    if len(data.position) != 7:
        rospy.logwarn("Número de juntas incorreto. Esperado: 7, Recebido: %d", len(data.position))
        return

    # 1. Envia o comando UR5 (6 juntas)
    ur5_command = JointTrajectory()
    ur5_command.joint_names = JOINT_NAMES
    point_ur5 = JointTrajectoryPoint()
    point_ur5.positions = data.position[:6]
    point_ur5.time_from_start = rospy.Duration(0.1)
    ur5_command.points.append(point_ur5)
    pub_ur5.publish(ur5_command)

    # 2. Envia o comando da Garra (1 junta)
    gripper_command = JointTrajectory()
    gripper_command.joint_names = [GRIPPER_JOINT_NAME]
    point_gripper = JointTrajectoryPoint()
    point_gripper.positions = data.position[6:]
    point_gripper.time_from_start = rospy.Duration(0.1)
    gripper_command.points.append(point_gripper)
    pub_gripper.publish(gripper_command)

    # 3. Atualiza o estado atual para o Reset (CRUCIAL!)
    latest_joint_state = data 
    rospy.loginfo(f"Valores de teleoperação recebidos e enviados. Pose atualizada.")


def reset_pose_callback(data):
    """
    Callback de Reset: Recebe a meta final do Unity e calcula a trajetória suave.
    """
    global latest_joint_state
    
    if latest_joint_state is None:
        rospy.logerr("Não é possível resetar: A pose inicial do robô é desconhecida. Aguarde a teleoperação iniciar.")
        return

    # A meta (posição final) é enviada do Unity (7 posições em RADIANOS)
    final_pose = data.data 
    initial_pose = latest_joint_state.position # Posição inicial (7 posições em radianos)
    
    # 1. Gera a JointTrajectory suave para as 7 juntas
    full_trajectory = generate_smooth_trajectory(initial_pose, final_pose, RESET_TIME_SECONDS, PUB_FREQUENCY_HZ)
    
    # 2. Separa a trajetória em comandos para o UR5 e para a Garra
    ur5_command = JointTrajectory()
    gripper_command = JointTrajectory()
    
    ur5_command.joint_names = JOINT_NAMES
    gripper_command.joint_names = [GRIPPER_JOINT_NAME]
    
    for point in full_trajectory.points:
        # Ponto para o UR5
        point_ur5 = JointTrajectoryPoint()
        point_ur5.positions = point.positions[:6]
        point_ur5.time_from_start = point.time_from_start
        ur5_command.points.append(point_ur5)
        
        # Ponto para a Garra
        point_gripper = JointTrajectoryPoint()
        point_gripper.positions = point.positions[6:] # Apenas o último valor
        point_gripper.time_from_start = point.time_from_start
        gripper_command.points.append(point_gripper)
        
    # 3. Publica ambas as trajetórias (Movimento Suave)
    pub_ur5.publish(ur5_command)
    pub_gripper.publish(gripper_command)
    rospy.loginfo(f"Trajetória suave ({RESET_TIME_SECONDS}s) enviada para o controlador.")

    # 4. Publicar a trajetória suave para o Unity para espelhamento
    pub_unity_traj.publish(full_trajectory)
    rospy.loginfo("Trajetória suave (Polinômio Quíntuplo) enviada para o controlador E para o Unity.")    



def main():
    """
    Configura o nó ROS, se inscreve nos tópicos e mantém o nó ativo.
    """
    # Subscriber principal para a teleoperação
    rospy.Subscriber('unity/joint_command', JointState, unity_joint_state_callback)
    
    # NOVO Subscriber para o comando de Reset
    rospy.Subscriber(RESET_GOAL_TOPIC, Float64MultiArray, reset_pose_callback)
    
    rospy.loginfo(f"Nó de controle HÍBRIDO iniciado. Escutando Teleop em: unity/joint_command e Reset em: {RESET_GOAL_TOPIC}")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
