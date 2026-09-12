#!/usr/bin/env python3
# Arquivo: real_quintic_planner_node.py
# Ajuste: Atualizado com a lógica validada no Gazebo (Carimbo de tempo e Desenrolador)
# Ajuste em 12/09/2026 às 20:54h
import rospy
import math
import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def resolver_salto_angular(angulo_alvo, angulo_atual):
    # Limites físicos de segurança do UR5 (±360 graus, que são ±2*Pi radianos)
    # Colocamos 6.28 como margem de segurança
    limite_inferior = -6.28
    limite_superior = 6.28
    
    # 1. Encontra a menor distância matemática (Pode violar os limites)
    diferenca = angulo_alvo - angulo_atual
    menor_distancia = math.atan2(math.sin(diferenca), math.cos(diferenca))
    alvo_proposto = angulo_atual + menor_distancia
    
    # 2. O DESENROLADOR: Se bater na parede física, obriga a dar a volta por trás
    while alvo_proposto > limite_superior:
        alvo_proposto -= 2 * math.pi
        
    while alvo_proposto < limite_inferior:
        alvo_proposto += 2 * math.pi
        
    return alvo_proposto

class QuinticPlannerNode:
    def __init__(self):
        rospy.init_node('quintic_planner_node')
        
        self.movement_duration = 3.0 
        self.hz = 50.0 
        
        rospy.Subscriber('/unity/target_joints', JointState, self.target_callback)
        
        # [ROBÔ REAL]: Lendo os motores físicos da raiz global
        rospy.Subscriber('/joint_states', JointState, self.current_state_callback)
        
        self.traj_pub = rospy.Publisher('/ur5/planned_trajectory', JointTrajectory, queue_size=1)
        
        self.current_joints = {}
        self.is_ready = False
        
        rospy.loginfo("Planejador Quíntuplo (Hardware Real) Iniciado...")

    def current_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            self.current_joints[name] = msg.position[i]
        self.is_ready = True

    def target_callback(self, msg):
        if not self.is_ready:
            rospy.logwarn("Aguardando posições reais do robô...")
            return

        joint_names = msg.name
        target_positions = msg.position
        
        trajectory = JointTrajectory()
        
        # --- CORREÇÃO VITAL: CARIMBO DE TEMPO ---
        # Sincroniza o relógio para o controlador do robô real não rejeitar a mensagem por atraso temporal
        trajectory.header.stamp = rospy.Time.now()
        trajectory.joint_names = joint_names
        
        num_points = int(self.movement_duration * self.hz)
        passo_tempo = self.movement_duration / num_points
        t_array = np.linspace(passo_tempo, self.movement_duration, num_points)
        
        for t in t_array:
            point = JointTrajectoryPoint()
            point.time_from_start = rospy.Duration(t)
            
            positions, velocities, accelerations = [], [], []
            
            for i, joint_name in enumerate(joint_names):
                # Agora self.current_joints tem a posição real medida pelos encoders físicos
                q0 = self.current_joints.get(joint_name, 0.0)
                qf_bruto = target_positions[i] 

                qf = resolver_salto_angular(qf_bruto, q0)
                
                T = self.movement_duration
                a0, a1, a2 = q0, 0.0, 0.0
                a3 = 10 * (qf - q0) / (T**3)
                a4 = -15 * (qf - q0) / (T**4)
                a5 = 6 * (qf - q0) / (T**5)
                
                pos = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)
                vel = a1 + 2*a2*t + 3*a3*(t**2) + 4*a4*(t**3) + 5*a5*(t**4)
                acc = 2*a2 + 6*a3*t + 12*a4*(t**2) + 20*a5*(t**3)
                
                positions.append(pos)
                velocities.append(vel)
                accelerations.append(acc)
                
            point.positions = positions
            point.velocities = velocities
            point.accelerations = accelerations
            trajectory.points.append(point)
            
        self.traj_pub.publish(trajectory)
        rospy.loginfo("Trajetória Suavizada e Sincronizada Publicada (Hardware Real)!")

if __name__ == '__main__':
    try:
        QuinticPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass















# #!/usr/bin/env python3
# # Arquivo: real_quintic_planner_node.py
# # Ajuste realizado em 10/09/2026, validado no gazebo
# import rospy
# import math
# import numpy as np
# from sensor_msgs.msg import JointState
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# def resolver_salto_angular(angulo_alvo, angulo_atual):
#     diferenca = angulo_alvo - angulo_atual
#     menor_distancia = math.atan2(math.sin(diferenca), math.cos(diferenca))
#     return angulo_atual + menor_distancia

# class QuinticPlannerNode:
#     def __init__(self):
#         rospy.init_node('quintic_planner_node')
        
#         self.movement_duration = 3.0 
#         self.hz = 50.0 
        
#         rospy.Subscriber('/unity/target_joints', JointState, self.target_callback)
        
#         # [ALTERAÇÃO 2]: Lendo os motores físicos da raiz
#         rospy.Subscriber('/joint_states', JointState, self.current_state_callback)
        
#         self.traj_pub = rospy.Publisher('/ur5/planned_trajectory', JointTrajectory, queue_size=1)
        
#         self.current_joints = {}
#         self.is_ready = False
        
#         rospy.loginfo("Planejador Quíntuplo (Hardware Real) Iniciado...")

#     def current_state_callback(self, msg):
#         for i, name in enumerate(msg.name):
#             self.current_joints[name] = msg.position[i]
#         self.is_ready = True

#     def target_callback(self, msg):
#         if not self.is_ready:
#             return

#         joint_names = msg.name
#         target_positions = msg.position
        
#         trajectory = JointTrajectory()
#         trajectory.joint_names = joint_names
        
#         num_points = int(self.movement_duration * self.hz)
#         passo_tempo = self.movement_duration / num_points
#         t_array = np.linspace(passo_tempo, self.movement_duration, num_points)
        
#         for t in t_array:
#             point = JointTrajectoryPoint()
#             point.time_from_start = rospy.Duration(t)
            
#             positions, velocities, accelerations = [], [], []
            
#             for i, joint_name in enumerate(joint_names):
#                 q0 = self.current_joints.get(joint_name, 0.0)
#                 qf_bruto = target_positions[i] 

#                 qf = resolver_salto_angular(qf_bruto, q0)
                
#                 T = self.movement_duration
#                 a0, a1, a2 = q0, 0.0, 0.0
#                 a3 = 10 * (qf - q0) / (T**3)
#                 a4 = -15 * (qf - q0) / (T**4)
#                 a5 = 6 * (qf - q0) / (T**5)
                
#                 pos = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)
#                 vel = a1 + 2*a2*t + 3*a3*(t**2) + 4*a4*(t**3) + 5*a5*(t**4)
#                 acc = 2*a2 + 6*a3*t + 12*a4*(t**2) + 20*a5*(t**3)
                
#                 positions.append(pos)
#                 velocities.append(vel)
#                 accelerations.append(acc)
                
#             point.positions = positions
#             point.velocities = velocities
#             point.accelerations = accelerations
#             trajectory.points.append(point)
            
#         self.traj_pub.publish(trajectory)
#         rospy.loginfo("Trajetória Suavizada Publicada (Hardware Real)!")

# if __name__ == '__main__':
#     try:
#         QuinticPlannerNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass














# #!/usr/bin/env python3
# # Arquivo: real_robot_commander_node_140.py
# import rospy
# import actionlib
# from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output

# class RobotCommanderNode:
#     def __init__(self):
#         rospy.init_node('robot_commander_node')

#         # --- 1. CONEXÃO VIA ACTION SERVER (Tópico oficial do laboratório) ---
#         action_topic = '/follow_joint_trajectory'
#         self.arm_client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
        
#         rospy.loginfo(f"Aguardando o Action Server do robô em: {action_topic} ...")
#         self.arm_client.wait_for_server()
#         rospy.loginfo("Conectado! O UR5 físico autorizou o envio de comandos.")

#         # Tópico da Garra
#         self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=1)

#         # Escuta o Planejador Matemático
#         rospy.Subscriber('/ur5/planned_trajectory', JointTrajectory, self.trajectory_callback)

#         self.arm_joints = [
#             "shoulder_pan_joint",
#             "shoulder_lift_joint",
#             "elbow_joint",
#             "wrist_1_joint",
#             "wrist_2_joint",
#             "wrist_3_joint"
#         ]
#         self.gripper_joint = "finger_joint"

#     def trajectory_callback(self, msg):
#         rospy.loginfo("Trajetória recebida! Contratando o Action Server para mover o robô...")

#         arm_msg = JointTrajectory()
#         arm_msg.header = Header()
#         arm_msg.header.stamp = rospy.Time.now()  # Carimbo de tempo vital!
#         arm_msg.header.frame_id = "base_link"
#         arm_msg.joint_names = self.arm_joints

#         try:
#             arm_indices = [msg.joint_names.index(j) for j in self.arm_joints]
#         except ValueError as e:
#             rospy.logerr(f"Erro: Faltam juntas do braço UR5. {e}")
#             return

#         has_gripper = False
#         try:
#             gripper_index = msg.joint_names.index(self.gripper_joint)
#             has_gripper = True
#         except ValueError:
#             rospy.logwarn("Aviso: Mensagem sem dados da garra.")

#         for point in msg.points:
#             arm_point = JointTrajectoryPoint()
#             arm_point.time_from_start = point.time_from_start
            
#             # Repassando posições, velocidades e acelerações calculadas pelo planejador
#             arm_point.positions = [point.positions[i] for i in arm_indices]
#             if point.velocities:
#                 arm_point.velocities = [point.velocities[i] for i in arm_indices]
#             if point.accelerations:
#                 arm_point.accelerations = [point.accelerations[i] for i in arm_indices]
                
#             arm_msg.points.append(arm_point)

#         # --- 2. ENVIANDO O OBJETIVO PARA O ROBÔ ---
#         goal = FollowJointTrajectoryGoal()
#         goal.trajectory = arm_msg
        
#         # Envia e não bloqueia o código (assíncrono)
#         self.arm_client.send_goal(goal)

#         # Roteamento da Garra Robotiq
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

#         rospy.loginfo("Sucesso! Contrato de movimento despachado para o UR5.")

# if __name__ == '__main__':
#     try:
#         RobotCommanderNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass








# #!/usr/bin/env python3
# # Arquivo: real_robot_quintic_planner_node_140.py, com correção do tempo zero para o robô real
# import rospy
# import numpy as np
# from sensor_msgs.msg import JointState
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# class QuinticPlannerNode:
#     def __init__(self):
#         rospy.init_node('quintic_planner_node')
        
#         # Parâmetros
#         self.movement_duration = 3.0 # Segundos
#         self.hz = 50.0 # Frequência dos pontos da trajetória (50 pontos por segundo)
        
#         # Ouve o Unity e o estado atual do Robô Real
#         rospy.Subscriber('/unity/target_joints', JointState, self.target_callback)
#         rospy.Subscriber('/joint_states', JointState, self.current_state_callback)
        
#         # Publica a trajetória planejada (O Comandante vai ler isso depois)
#         self.traj_pub = rospy.Publisher('/ur5/planned_trajectory', JointTrajectory, queue_size=1)
        
#         self.current_joints = {}
#         self.is_ready = False
        
#         rospy.loginfo("Planejador Quíntuplo Iniciado. Aguardando comandos do Unity...")

#     def current_state_callback(self, msg):
#         # Atualiza a memória de onde o robô está agora
#         for i, name in enumerate(msg.name):
#             self.current_joints[name] = msg.position[i]
#         self.is_ready = True

#     def target_callback(self, msg):
#         if not self.is_ready:
#             rospy.logwarn("Ainda não recebi o estado atual do robô. Ignorando comando.")
#             return

#         rospy.loginfo("Alvo autônomo recebido do Unity. Calculando trajetória quíntupla...")
        
#         joint_names = msg.name
#         target_positions = msg.position
        
#         # Cria a mensagem de trajetória
#         trajectory = JointTrajectory()
#         trajectory.joint_names = joint_names
        
#         # --- CORREÇÃO DO TEMPO ZERO PARA O ROBÔ REAL ---
#         # Calcula o número de pontos
#         num_points = int(self.movement_duration * self.hz)
        
#         # Gera o tempo começando do primeiro "passo" (ex: 0.02s), evitando o t=0.0
#         passo_tempo = self.movement_duration / num_points
#         t_array = np.linspace(passo_tempo, self.movement_duration, num_points)
#         # -----------------------------------------------
        
#         # Para cada instante de tempo, criamos um ponto na trajetória
#         for t in t_array:
#             point = JointTrajectoryPoint()
#             point.time_from_start = rospy.Duration(t)
            
#             positions = []
#             velocities = []
#             accelerations = []
            
#             # Calcula o polinômio para cada junta individualmente
#             for i, joint_name in enumerate(joint_names):
#                 q0 = self.current_joints.get(joint_name, 0.0)
#                 qf = target_positions[i]
                
#                 # Coeficientes do polinômio quíntuplo (Condições de contorno: v0=vf=0, a0=af=0)
#                 T = self.movement_duration
#                 a0 = q0
#                 a1 = 0.0
#                 a2 = 0.0
#                 a3 = 10 * (qf - q0) / (T**3)
#                 a4 = -15 * (qf - q0) / (T**4)
#                 a5 = 6 * (qf - q0) / (T**5)
                
#                 # Equações cinemáticas
#                 pos = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)
#                 vel = a1 + 2*a2*t + 3*a3*(t**2) + 4*a4*(t**3) + 5*a5*(t**4)
#                 acc = 2*a2 + 6*a3*t + 12*a4*(t**2) + 20*a5*(t**3)
                
#                 positions.append(pos)
#                 velocities.append(vel)
#                 accelerations.append(acc)
                
#             point.positions = positions
#             point.velocities = velocities
#             point.accelerations = accelerations
#             trajectory.points.append(point)
            
#         # Publica a obra de arte matemática
#         self.traj_pub.publish(trajectory)
#         rospy.loginfo("Trajetória Suavizada Publicada para o hardware!")

# if __name__ == '__main__':
#     try:
#         QuinticPlannerNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass















# #!/usr/bin/env python3
# # Arquivo: quintic_planner_node_140.py
# import rospy
# import numpy as np
# from sensor_msgs.msg import JointState
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# class QuinticPlannerNode:
#     def __init__(self):
#         rospy.init_node('quintic_planner_node')
        
#         # Parâmetros
#         self.movement_duration = 3.0 # Segundos
#         self.hz = 50.0 # Frequência dos pontos da trajetória (50 pontos por segundo)
        
#         # Ouve o Unity e o estado atual do Gazebo
#         rospy.Subscriber('/unity/target_joints', JointState, self.target_callback)
#         rospy.Subscriber('/joint_states', JointState, self.current_state_callback)
        
#         # Publica a trajetória planejada (O Comandante vai ler isso depois)
#         self.traj_pub = rospy.Publisher('/ur5/planned_trajectory', JointTrajectory, queue_size=1)
        
#         self.current_joints = {}
#         self.is_ready = False
        
#         rospy.loginfo("Planejador Quíntuplo Iniciado. Aguardando comandos do Unity...")

#     def current_state_callback(self, msg):
#         # Atualiza a memória de onde o robô está agora
#         for i, name in enumerate(msg.name):
#             self.current_joints[name] = msg.position[i]
#         self.is_ready = True

#     def target_callback(self, msg):
#         if not self.is_ready:
#             rospy.logwarn("Ainda não recebi o estado atual do robô. Ignorando comando.")
#             return

#         rospy.loginfo("Alvo recebido do Unity. Calculando trajetória quíntupla...")
        
#         joint_names = msg.name
#         target_positions = msg.position
        
#         # Cria a mensagem de trajetória
#         trajectory = JointTrajectory()
#         trajectory.joint_names = joint_names
        
#         # Gera o tempo
#         t_array = np.linspace(0, self.movement_duration, int(self.movement_duration * self.hz))
        
#         # Para cada instante de tempo, criamos um ponto na trajetória
#         for t in t_array:
#             point = JointTrajectoryPoint()
#             point.time_from_start = rospy.Duration(t)
            
#             positions = []
#             velocities = []
#             accelerations = []
            
#             # Calcula o polinômio para cada junta individualmente
#             for i, joint_name in enumerate(joint_names):
#                 q0 = self.current_joints.get(joint_name, 0.0)
#                 qf = target_positions[i]
                
#                 # Coeficientes do polinômio quíntuplo (Condições de contorno: v0=vf=0, a0=af=0)
#                 T = self.movement_duration
#                 a0 = q0
#                 a1 = 0.0
#                 a2 = 0.0
#                 a3 = 10 * (qf - q0) / (T**3)
#                 a4 = -15 * (qf - q0) / (T**4)
#                 a5 = 6 * (qf - q0) / (T**5)
                
#                 # Equações cinemáticas
#                 pos = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)
#                 vel = a1 + 2*a2*t + 3*a3*(t**2) + 4*a4*(t**3) + 5*a5*(t**4)
#                 acc = 2*a2 + 6*a3*t + 12*a4*(t**2) + 20*a5*(t**3)
                
#                 positions.append(pos)
#                 velocities.append(vel)
#                 accelerations.append(acc)
                
#             point.positions = positions
#             point.velocities = velocities
#             point.accelerations = accelerations
#             trajectory.points.append(point)
            
#         # Publica a obra de arte matemática
#         self.traj_pub.publish(trajectory)
#         rospy.loginfo("Trajetória Suavizada Publicada!")

# if __name__ == '__main__':
#     try:
#         QuinticPlannerNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass
