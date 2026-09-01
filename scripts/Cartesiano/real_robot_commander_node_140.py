#!/usr/bin/env python3
# Arquivo: real_robot_commander_node_140.py
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output

class RobotCommanderNode:
    def __init__(self):
        rospy.init_node('robot_commander_node')

        # Tópicos de Saída (Hardware Real)
        self.arm_pub = rospy.Publisher('/scaled_pos_joint_traj_controller/command', JointTrajectory, queue_size=1)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=1)

        # Tópico de Entrada (Vindo do Planejador Matemático)
        rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

        # Dicionário de Juntas do Braço e Garra
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
        rospy.loginfo("Trajetória Quíntupla recebida! Processando comandos para o robô real...")

        # Prepara a mensagem do Braço
        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joints

        # --- 1. IDENTIFICAÇÃO SEGURA DOS ÍNDICES DAS JUNTAS ---
        try:
            arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
        except ValueError as e:
            rospy.logerr(f"Erro: Faltam juntas do braço UR5 na mensagem do Planejador. {e}")
            return

        # Verifica se a garra está presente na mensagem sem travar se estiver ausente
        has_gripper = False
        try:
            gripper_index = msg.joint_names.index(self.gripper_joint)
            has_gripper = True
        except ValueError:
            rospy.logwarn("Aviso: Mensagem sem dados da garra (finger_joint). Executando apenas a trajetória do braço.")

        # --- 2. ROTEAMENTO DO BRAÇO (Trajetória Completa) ---
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

        # Envia a trajetória para o controlador oficial do UR5
        self.arm_pub.publish(arm_msg)

        # --- 3. ROTEAMENTO DA GARRA (Se presente) ---
        if has_gripper and len(msg.points) > 0:
            final_point = msg.points[-1]
            gripper_val = final_point.positions[gripper_index]
            rpr_val = int(max(0, min(255, (gripper_val / 0.8) * 255.0)))

            gripper_msg = Robotiq2FGripper_robot_output()
            gripper_msg.rACT = 1
            gripper_msg.rGTO = 1
            gripper_msg.rATR = 0
            gripper_msg.rPR = rpr_val
            gripper_msg.rSP = 150
            gripper_msg.rFR = 150

            self.gripper_pub.publish(gripper_msg)

        rospy.loginfo("Sucesso! Comandos despachados para o UR5 físico.")

if __name__ == '__main__':
    try:
        RobotCommanderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


















# #!/usr/bin/env python3
# # Arquivo: robot_commander_node_140.py
# import rospy
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output # NOVO

# class RobotCommanderNode:
#     def __init__(self):
#         rospy.init_node('robot_commander_node')

#         # Tópicos de Saída
#         self.arm_pub = rospy.Publisher('/scaled_pos_joint_traj_controller/command', JointTrajectory, queue_size=1)
#         self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=1)

#         # Tópico de Entrada (Vindo do Planejador Matemático)
#         rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

#         # Dicionário de Juntas para separar o Joio do Trigo
#         self.arm_joints = [
#             "shoulder_pan_joint",
#             "shoulder_lift_joint",
#             "elbow_joint",
#             "wrist_1_joint",
#             "wrist_2_joint",
#             "wrist_3_joint"
#         ]
#         self.gripper_joint = "finger_joint"

#         rospy.loginfo("Comandante do Robô Iniciado. Aguardando rotas suavizadas do Planejador...")

#     def trajectory_callback(self, msg):
#         rospy.loginfo("Trajetória Quíntupla recebida! Roteando comandos para o Gazebo...")

#         # Prepara a mensagem do Braço
#         arm_msg = JointTrajectory()
#         arm_msg.joint_names = self.arm_joints

#         # Prepara a mensagem da Garra
#         gripper_msg = JointTrajectory()
#         gripper_msg.joint_names = [self.gripper_joint]

#         try:
#             # Descobre em qual posição do Array estão as juntas que queremos
#             arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
#             gripper_index = msg.joint_names.index(self.gripper_joint)
#         except ValueError as e:
#             rospy.logerr(f"Erro: O Planejador enviou nomes de juntas que eu não reconheço. {e}")
#             return

#         # --- 1. ROTEAMENTO DO BRAÇO (Trajetória Completa) ---
#         for point in msg.points:
#             arm_point = JointTrajectoryPoint()
#             arm_point.time_from_start = point.time_from_start
            
#             # Extrai apenas os valores das 6 juntas do braço
#             arm_point.positions = [point.positions[i] for i in arm_indices]
#             if point.velocities:
#                 arm_point.velocities = [point.velocities[i] for i in arm_indices]
#             if point.accelerations:
#                 arm_point.accelerations = [point.accelerations[i] for i in arm_indices]
                
#             arm_msg.points.append(arm_point)

#         # --- 2. ROTEAMENTO DA GARRA (Convertendo para Hardware Físico) ---
#         final_point = msg.points[-1]
#         gripper_val = final_point.positions[gripper_index]
#         rpr_val = int(max(0, min(255, (gripper_val / 0.8) * 255.0)))

#         gripper_msg = Robotiq2FGripper_robot_output()
#         gripper_msg.rACT = 1
#         gripper_msg.rGTO = 1
#         gripper_msg.rATR = 0
#         gripper_msg.rPR = rpr_val
#         gripper_msg.rSP = 150
#         gripper_msg.rFR = 150


#         # --- 3. EXECUÇÃO ---
#         self.arm_pub.publish(arm_msg)
#         self.gripper_pub.publish(gripper_msg)
        
#         rospy.loginfo("Sucesso! Comandos despachados para os controladores do Gazebo.")

# if __name__ == '__main__':
#     try:
#         RobotCommanderNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass
