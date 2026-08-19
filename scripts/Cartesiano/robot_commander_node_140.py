#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class RobotCommanderNode:
    def __init__(self):
        rospy.init_node('robot_commander_node')

        # Tópicos de Saída (Direto para o Gazebo)
        self.arm_pub = rospy.Publisher('/ur5/eff_joint_traj_controller/command', JointTrajectory, queue_size=1)
        self.gripper_pub = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=1)

        # Tópico de Entrada (Vindo do Planejador Matemático)
        rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

        # Dicionário de Juntas para separar o Joio do Trigo
        self.arm_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        self.gripper_joint = "finger_joint"

        rospy.loginfo("Comandante do Robô Iniciado. Aguardando rotas suavizadas do Planejador...")

    def trajectory_callback(self, msg):
        rospy.loginfo("Trajetória Quíntupla recebida! Roteando comandos para o Gazebo...")

        # Prepara a mensagem do Braço
        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joints

        # Prepara a mensagem da Garra
        gripper_msg = JointTrajectory()
        gripper_msg.joint_names = [self.gripper_joint]

        try:
            # Descobre em qual posição do Array estão as juntas que queremos
            arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
            gripper_index = msg.joint_names.index(self.gripper_joint)
        except ValueError as e:
            rospy.logerr(f"Erro: O Planejador enviou nomes de juntas que eu não reconheço. {e}")
            return

        # --- 1. ROTEAMENTO DO BRAÇO (Trajetória Completa) ---
        for point in msg.points:
            arm_point = JointTrajectoryPoint()
            arm_point.time_from_start = point.time_from_start
            
            # Extrai apenas os valores das 6 juntas do braço
            arm_point.positions = [point.positions[i] for i in arm_indices]
            if point.velocities:
                arm_point.velocities = [point.velocities[i] for i in arm_indices]
            if point.accelerations:
                arm_point.accelerations = [point.accelerations[i] for i in arm_indices]
                
            arm_msg.points.append(arm_point)

        # --- 2. ROTEAMENTO DA GARRA (Apenas o destino final) ---
        # A garra não precisa de uma curva quíntupla suave, ela só precisa saber se abre ou fecha.
        # Então pegamos apenas o último ponto da trajetória gerada.
        final_point = msg.points[-1]
        gripper_point = JointTrajectoryPoint()
        gripper_point.time_from_start = rospy.Duration(1.0) # Tempo de acionamento da garra
        gripper_point.positions = [final_point.positions[gripper_index]]
        gripper_msg.points.append(gripper_point)

        # --- 3. EXECUÇÃO ---
        self.arm_pub.publish(arm_msg)
        self.gripper_pub.publish(gripper_msg)
        
        rospy.loginfo("Sucesso! Comandos despachados para os controladores do Gazebo.")

if __name__ == '__main__':
    try:
        RobotCommanderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
