#!/usr/bin/env python3
# Arquivo: real_robot_commander_node.py
# Ajuste: Roteamento Híbrido (VR/Autônomo) integrado com a Garra Robotiq Física
# Ambiente: HARDWARE REAL UR5
# Ajuste realizado em 12/09/2026, às 20:57h

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Importando a biblioteca específica da sua garra física Robotiq
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output

class RobotCommanderNode:
    def __init__(self):
        rospy.init_node('robot_commander_node')

        # =========================================================
        # 1. PISTA BUROCRÁTICA (Action Server - MODO AUTÔNOMO)
        # =========================================================
        # Tópico padrão do ur_robot_driver para execução segura de trajetórias
        action_topic = '/scaled_pos_joint_traj_controller/follow_joint_trajectory'
        self.arm_client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
        
        rospy.loginfo(f"Aguardando o Action Server do robô real em: {action_topic} ...")
        self.arm_client.wait_for_server()
        rospy.loginfo("Conectado! O UR5 físico autorizou o envio de comandos.")

        # =========================================================
        # 2. PISTA EXPRESSA (Publisher - MODO VR)
        # =========================================================
        # Streaming direto para evitar 'gagueira' no rastreamento contínuo
        pub_topic = '/scaled_pos_joint_traj_controller/command'
        self.arm_pub = rospy.Publisher(pub_topic, JointTrajectory, queue_size=1)

        # =========================================================
        # 3. GARRA FÍSICA (Robotiq)
        # =========================================================
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=1)

        # Assina o tópico unificado (KDL ou Quintic Planner enviam para cá)
        rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

        self.arm_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.gripper_joint = "finger_joint"

        self.current_mode = "NENHUM"
        rospy.loginfo("Comandante Físico Híbrido Iniciado! Roteamento Dinâmico Ativado.")

    def trajectory_callback(self, msg):
        num_points = len(msg.points)
        if num_points == 0:
            return

        arm_msg = JointTrajectory()
        # Preserva o relógio original (vital para o driver real não rejeitar o comando)
        arm_msg.header = msg.header
        arm_msg.joint_names = self.arm_joints

        try:
            arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
        except ValueError as e:
            rospy.logdebug(f"Aviso Commander: Mensagem ignorada pelo braço. Motivo: {e}")
            return

        has_gripper = False
        try:
            gripper_index = msg.joint_names.index(self.gripper_joint)
            has_gripper = True
        except ValueError:
            pass

        for point in msg.points:
            arm_point = JointTrajectoryPoint()
            arm_point.time_from_start = point.time_from_start
            arm_point.positions = [point.positions[i] for i in arm_indices]

            # Repassa velocidades e acelerações com segurança
            if point.velocities:
                arm_point.velocities = [point.velocities[i] for i in arm_indices]
            else:
                arm_point.velocities = [0.0] * 6
                
            if point.accelerations:
                arm_point.accelerations = [point.accelerations[i] for i in arm_indices]
            else:
                arm_point.accelerations = [0.0] * 6

            arm_msg.points.append(arm_point)

        # =========================================================
        # O GUARDA DE TRÂNSITO (ROTEAMENTO INTELIGENTE)
        # =========================================================
        if num_points > 1:
            # MODO AUTÔNOMO: Trajetória gerada pelo Quintic Planner (Múltiplos pontos)
            if self.current_mode != "AUTONOMO":
                rospy.loginfo(">>> MODO AUTÔNOMO DETECTADO: Roteando via Action Server...")
                self.current_mode = "AUTONOMO"
            
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = arm_msg
            self.arm_client.send_goal(goal)
            
        else:
            # MODO VR: Streaming direto da Mão (1 ponto por vez)
            if self.current_mode != "VR":
                rospy.loginfo(">>> MODO VR DETECTADO: Cancelando rotinas e ativando Streaming Direto...")
                # A CEREJA DO BOLO: Freia o braço se ele estava em viagem automática
                self.arm_client.cancel_all_goals() 
                self.current_mode = "VR"
                
            self.arm_pub.publish(arm_msg)

        # =========================================================
        # CONTROLE DA GARRA ROBOTIQ FÍSICA
        # =========================================================
        if has_gripper and num_points > 0:
            final_point = msg.points[-1]
            gripper_val = final_point.positions[gripper_index]
            
            # Tradução do valor (0.0 a 0.8 radianos) para o limite da placa (0 a 255 bytes)
            rpr_val = int(max(0, min(255, (gripper_val / 0.8) * 255.0)))

            gripper_msg = Robotiq2FGripper_robot_output()
            gripper_msg.rACT = 1     # Ativação
            gripper_msg.rGTO = 1     # Go-To (Executar movimento)
            gripper_msg.rATR = 0     # Auto-Release
            gripper_msg.rPR = rpr_val  # Position Request (Abertura real calculada)
            gripper_msg.rSP = 150    # Velocidade (Speed)
            gripper_msg.rFR = 150    # Força (Force)
            
            self.gripper_pub.publish(gripper_msg)

if __name__ == '__main__':
    try:
        RobotCommanderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass















# #!/usr/bin/env python3
# # Arquivo: real_robot_commander_node.py
# # Ajuste realizado em 10/09/2026, validado no gazebo
# import rospy
# import actionlib
# from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header

# # [ALTERAÇÃO 3]: Importando a biblioteca específica da sua garra física
# from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output

# class RobotCommanderNode:
#     def __init__(self):
#         rospy.init_node('robot_commander_node')

#         # --- MÚSCULOS DO ROBÔ FÍSICO ---
#         # [ALTERAÇÃO 4]: Conectando ao Action Server oficial da UR5
#         action_topic = '/follow_joint_trajectory'
#         self.arm_client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
        
#         rospy.loginfo(f"Aguardando o Action Server do robô em: {action_topic} ...")
#         self.arm_client.wait_for_server()
#         rospy.loginfo("Conectado! O UR5 físico autorizou o envio de comandos.")

#         # [ALTERAÇÃO 5]: Publicador oficial da placa da Robotiq 2F
#         self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=1)

#         rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

#         self.arm_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
#                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#         self.gripper_joint = "finger_joint"

#         rospy.loginfo("Comandante Físico Iniciado. Lógica de segurança purificada.")

#     def trajectory_callback(self, msg):
#         num_points = len(msg.points)

#         arm_msg = JointTrajectory()
#         arm_msg.header = Header()
#         arm_msg.header.stamp = rospy.Time.now()
#         arm_msg.header.frame_id = "base_link"
#         arm_msg.joint_names = self.arm_joints

#         try:
#             arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
#         except ValueError as e:
#             rospy.logdebug(f"Aviso Commander: Mensagem ignorada pelo braço. Motivo: {e}")
#             return

#         has_gripper = False
#         try:
#             gripper_index = msg.joint_names.index(self.gripper_joint)
#             has_gripper = True
#         except ValueError:
#             pass

#         for point in msg.points:
#             arm_point = JointTrajectoryPoint()
#             arm_point.time_from_start = point.time_from_start
#             arm_point.positions = [point.positions[i] for i in arm_indices]
            
#             if point.velocities:
#                 arm_point.velocities = [point.velocities[i] for i in arm_indices]
#             else:
#                 arm_point.velocities = [0.0] * 6
                
#             if point.accelerations:
#                 arm_point.accelerations = [point.accelerations[i] for i in arm_indices]
#             else:
#                 arm_point.accelerations = [0.0] * 6
                
#             arm_msg.points.append(arm_point)

#         # --- ENVIO PARA O BRAÇO VIA CONTRATO ---
#         goal = FollowJointTrajectoryGoal()
#         goal.trajectory = arm_msg
#         self.arm_client.send_goal(goal)

#         # --- ENVIO PARA A GARRA ROBOTIQ FÍSICA ---
#         if has_gripper and num_points > 0:
#             final_point = msg.points[-1]
#             gripper_val = final_point.positions[gripper_index]
            
#             # [ALTERAÇÃO 6]: Tradução do ângulo (radianos) para o limite de força da garra real (0 a 255)
#             rpr_val = int(max(0, min(255, (gripper_val / 0.8) * 255.0)))

#             gripper_msg = Robotiq2FGripper_robot_output()
#             gripper_msg.rACT = 1
#             gripper_msg.rGTO = 1
#             gripper_msg.rATR = 0
#             gripper_msg.rPR = rpr_val
#             gripper_msg.rSP = 150
#             gripper_msg.rFR = 150
#             self.gripper_pub.publish(gripper_msg)

# if __name__ == '__main__':
#     try:
#         RobotCommanderNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass









# #!/usr/bin/env python3
# # Arquivo: real_robot_commander_node_140.py
# import rospy
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output

# class RobotCommanderNode:
#     def __init__(self):
#         rospy.init_node('robot_commander_node')

#         # Tópicos de Saída (Hardware Real)
#         self.arm_pub = rospy.Publisher('/scaled_pos_joint_traj_controller/command', JointTrajectory, queue_size=1)
#         self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=1)

#         # Tópico de Entrada (Vindo do Planejador Matemático)
#         rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

#         # Dicionário de Juntas do Braço e Garra
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
#         rospy.loginfo("Trajetória Quíntupla recebida! Processando comandos para o robô real...")

#         # Prepara a mensagem do Braço
#         arm_msg = JointTrajectory()
#         arm_msg.joint_names = self.arm_joints

#         # --- 1. IDENTIFICAÇÃO SEGURA DOS ÍNDICES DAS JUNTAS ---
#         try:
#             arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
#         except ValueError as e:
#             rospy.logerr(f"Erro: Faltam juntas do braço UR5 na mensagem do Planejador. {e}")
#             return

#         # Verifica se a garra está presente na mensagem sem travar se estiver ausente
#         has_gripper = False
#         try:
#             gripper_index = msg.joint_names.index(self.gripper_joint)
#             has_gripper = True
#         except ValueError:
#             rospy.logwarn("Aviso: Mensagem sem dados da garra (finger_joint). Executando apenas a trajetória do braço.")

#         # --- 2. ROTEAMENTO DO BRAÇO (Trajetória Completa) ---
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

#         # Envia a trajetória para o controlador oficial do UR5
#         self.arm_pub.publish(arm_msg)

#         # --- 3. ROTEAMENTO DA GARRA (Se presente) ---
#         if has_gripper and len(msg.points) > 0:
#             final_point = msg.points[-1]
#             gripper_val = final_point.positions[gripper_index]
#             rpr_val = int(max(0, min(255, (gripper_val / 0.8) * 255.0)))

#             gripper_msg = Robotiq2FGripper_robot_output()
#             gripper_msg.rACT = 1
#             gripper_msg.rGTO = 1
#             gripper_msg.rATR = 0
#             gripper_msg.rPR = rpr_val
#             gripper_msg.rSP = 150
#             gripper_msg.rFR = 150

#             self.gripper_pub.publish(gripper_msg)

#         rospy.loginfo("Sucesso! Comandos despachados para o UR5 físico.")

# if __name__ == '__main__':
#     try:
#         RobotCommanderNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass


















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
