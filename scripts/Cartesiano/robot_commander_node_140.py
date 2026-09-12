#!/usr/bin/env python3
# Arquivo: robot_commander_node_140.py (GAZEBO HÍBRIDO - V1 SEGURA)
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class RobotCommanderNode:
    def __init__(self):
        rospy.init_node('robot_commander_node')

        # 1. CONEXÃO DA PISTA BUROCRÁTICA (Action Server - Menu)
        action_topic = '/ur5/eff_joint_traj_controller/follow_joint_trajectory'
        self.arm_client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
        rospy.loginfo("Aguardando Action Server no Gazebo...")
        self.arm_client.wait_for_server()

        # 2. CONEXÃO DA PISTA EXPRESSA (Publisher - VR)
        self.arm_pub = rospy.Publisher('/ur5/eff_joint_traj_controller/command', JointTrajectory, queue_size=1)
        
        # Garra (mantém via Publisher contínuo)
        self.gripper_pub = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=1)

        rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

        self.arm_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.gripper_joint = "finger_joint"
        
        self.current_mode = "NENHUM"
        rospy.loginfo("Comandante Híbrido Iniciado! Roteamento Dinâmico Ativado.")

    def trajectory_callback(self, msg):
        num_points = len(msg.points)
        if num_points == 0:
            return

        arm_msg = JointTrajectory()
        # Preserva o relógio original exato do Planejador ou do Unity
        arm_msg.header = msg.header
        arm_msg.joint_names = self.arm_joints

        try:
            arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
        except ValueError:
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

            # Repassa as derivadas do Polinômio para a física do Gazebo não estourar
            if len(point.velocities) > 0:
                arm_point.velocities = [point.velocities[i] for i in arm_indices]
            if len(point.accelerations) > 0:
                arm_point.accelerations = [point.accelerations[i] for i in arm_indices]

            arm_msg.points.append(arm_point)

        # =========================================================
        # O GUARDA DE TRÂNSITO (ROTEAMENTO INTELIGENTE)
        # =========================================================
        if num_points > 1:
            # MODO AUTÔNOMO: Trajetória gerada pelo Quintic Planner
            if self.current_mode != "AUTONOMO":
                rospy.loginfo(">>> MODO AUTÔNOMO DETECTADO: Usando Action Server...")
                self.current_mode = "AUTONOMO"
            
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = arm_msg
            self.arm_client.send_goal(goal)
            
        else:
            # MODO VR: Streaming direto da Mão (1 ponto a 90Hz)
            if self.current_mode != "VR":
                rospy.loginfo(">>> MODO VR DETECTADO: Usando Streaming (Publisher)...")
                self.arm_client.cancel_all_goals() # Freia qualquer trajetória autônoma rodando
                self.current_mode = "VR"
                
            self.arm_pub.publish(arm_msg)

        # =========================================================
        # CONTROLE DA GARRA
        # =========================================================
        if has_gripper and num_points > 0:
            gripper_msg = JointTrajectory(joint_names=[self.gripper_joint])
            gripper_msg.header = msg.header
            gripper_point = JointTrajectoryPoint()
            gripper_point.positions = [msg.points[-1].positions[gripper_index]]
            gripper_point.time_from_start = msg.points[-1].time_from_start
            gripper_msg.points.append(gripper_point)
            self.gripper_pub.publish(gripper_msg)

if __name__ == '__main__':
    try:
        RobotCommanderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass











# #!/usr/bin/env python3
# # Arquivo: robot_commander_node_140.py (GAZEBO OTIMIZADO PARA STREAMING)
# import rospy
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header

# class RobotCommanderNode:
#     def __init__(self):
#         rospy.init_node('robot_commander_node')

#         # --- O "MÚSCULO" VOLTA A SER STREAMING DIRETO PARA EVITAR SPAM DO ACTIONLIB ---
#         self.arm_pub = rospy.Publisher('/ur5/eff_joint_traj_controller/command', JointTrajectory, queue_size=1)
#         self.gripper_pub = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=1)

#         rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

#         self.arm_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
#                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#         self.gripper_joint = "finger_joint"
        
#         # Semáforo de Prioridade (Lógica idêntica ao real mantida)
#         self.autonomous_end_time = rospy.Time.now()

#         rospy.loginfo("Comandante do Gazebo Iniciado. Lógica de segurança real com streaming contínuo.")

#     def trajectory_callback(self, msg):
#         num_points = len(msg.points)

#         arm_msg = JointTrajectory()
#         arm_msg.header = Header()
#         arm_msg.header.stamp = rospy.Time.now()
#         arm_msg.header.frame_id = "base_link"
#         arm_msg.joint_names = self.arm_joints

#         # Identificação Segura do Braço
#         try:
#             arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
#         except ValueError as e:
#             # O linter vai gostar disso: estamos capturando e registrando o erro
#             rospy.logdebug(f"Aviso KDL: Mensagem ignorada pelo braço. Motivo: {e}")
#             return

#         # Identificação Segura da Garra
#         has_gripper = False
#         try:
#             gripper_index = msg.joint_names.index(self.gripper_joint)
#             has_gripper = True
#         except ValueError as e:
#             # Silencia no terminal para não dar spam, mas o 'as e' resolve o aviso do editor
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

#         # --- PUBLICAÇÃO DIRETA (Bypass no gargalo do Action Server) ---
#         self.arm_pub.publish(arm_msg)

#         if has_gripper and num_points > 0:
#             final_point = msg.points[-1]
#             gripper_msg = JointTrajectory()
#             gripper_msg.joint_names = [self.gripper_joint]
            
#             gripper_point = JointTrajectoryPoint()
#             gripper_point.time_from_start = rospy.Duration(1.0)
#             gripper_point.positions = [final_point.positions[gripper_index]]
#             gripper_msg.points.append(gripper_point)
            
#             self.gripper_pub.publish(gripper_msg)

# if __name__ == '__main__':
#     try:
#         RobotCommanderNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass










# #!/usr/bin/env python3
# # Arquivo: robot_commander_node_140.py (GAZEBO - 100% Fidedigno)
# import rospy
# import actionlib
# from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header

# class RobotCommanderNode:
#     def __init__(self):
#         rospy.init_node('robot_commander_node')

#         # --- 1. O "MÚSCULO" NO GAZEBO AGORA É UM ACTION CLIENT ---
#         action_topic = '/ur5/eff_joint_traj_controller/follow_joint_trajectory'
#         self.arm_client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
        
#         rospy.loginfo(f"Aguardando o Action Server do Gazebo em: {action_topic} ...")
#         self.arm_client.wait_for_server()
#         rospy.loginfo("Conectado! O Gazebo autorizou o envio de contratos.")

#         # A garra no Gazebo (Mantemos via Publisher, pois o gripper_controller costuma aceitar tópicos diretos)
#         self.gripper_pub = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=1)

#         rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

#         self.arm_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
#                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
#         self.gripper_joint = "finger_joint"
        
#         # Semáforo de Prioridade
#         self.autonomous_end_time = rospy.Time.now()

#     def trajectory_callback(self, msg):
#         num_points = len(msg.points)
        
#         # Lógica do Semáforo
#         if num_points > 1:
#             self.autonomous_end_time = rospy.Time.now() + rospy.Duration(3.5)
#         else:
#             if rospy.Time.now() < self.autonomous_end_time:
#                 return

#         arm_msg = JointTrajectory()
#         arm_msg.header = Header()
#         arm_msg.header.stamp = rospy.Time.now()
#         arm_msg.header.frame_id = "base_link"
#         arm_msg.joint_names = self.arm_joints

#         try:
#             arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
#         except ValueError:
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

#         # --- 2. ENVIANDO O CONTRATO PARA O GAZEBO ---
#         goal = FollowJointTrajectoryGoal()
#         goal.trajectory = arm_msg
        
#         self.arm_client.send_goal(goal)

#         if has_gripper and num_points > 0:
#             final_point = msg.points[-1]
#             gripper_msg = JointTrajectory()
#             gripper_msg.joint_names = [self.gripper_joint]
            
#             gripper_point = JointTrajectoryPoint()
#             gripper_point.time_from_start = rospy.Duration(1.0)
#             gripper_point.positions = [final_point.positions[gripper_index]]
#             gripper_msg.points.append(gripper_point)
            
#             self.gripper_pub.publish(gripper_msg)

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

# class RobotCommanderNode:
#     def __init__(self):
#         rospy.init_node('robot_commander_node')

#         # Tópicos de Saída (Direto para o Gazebo)
#         self.arm_pub = rospy.Publisher('/ur5/eff_joint_traj_controller/command', JointTrajectory, queue_size=1)
#         self.gripper_pub = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=1)

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

#         # --- 2. ROTEAMENTO DA GARRA (Apenas o destino final) ---
#         # A garra não precisa de uma curva quíntupla suave, ela só precisa saber se abre ou fecha.
#         # Então pegamos apenas o último ponto da trajetória gerada.
#         final_point = msg.points[-1]
#         gripper_point = JointTrajectoryPoint()
#         gripper_point.time_from_start = rospy.Duration(1.0) # Tempo de acionamento da garra
#         gripper_point.positions = [final_point.positions[gripper_index]]
#         gripper_msg.points.append(gripper_point)

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
