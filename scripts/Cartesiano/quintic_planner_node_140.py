#!/usr/bin/env python3
# Arquivo: quintic_planner_node_140.py (GAZEBO)
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
        rospy.Subscriber('/ur5/joint_states', JointState, self.current_state_callback)
        
        self.traj_pub = rospy.Publisher('/ur5/planned_trajectory', JointTrajectory, queue_size=1)
        
        self.current_joints = {}
        self.is_ready = False
        
        rospy.loginfo("Planejador Quíntuplo (Gazebo) Iniciado...")

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
        # Sincroniza o relógio para o Action Server não achar que a mensagem está atrasada
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
                # Agora, sem o nó falso, self.current_joints terá a posição real do Gazebo
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
        rospy.loginfo("Trajetória Suavizada e Sincronizada Publicada!")

if __name__ == '__main__':
    try:
        QuinticPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass













# #!/usr/bin/env python3
# # Arquivo: quintic_planner_node_140.py (GAZEBO)
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
#         rospy.Subscriber('/ur5/joint_states', JointState, self.current_state_callback)
        
#         self.traj_pub = rospy.Publisher('/ur5/planned_trajectory', JointTrajectory, queue_size=1)
        
#         self.current_joints = {}
#         self.is_ready = False
        
#         rospy.loginfo("Planejador Quíntuplo (Gazebo) Iniciado...")

#     def current_state_callback(self, msg):
#         for i, name in enumerate(msg.name):
#             self.current_joints[name] = msg.position[i]
#         self.is_ready = True

#     def target_callback(self, msg):
#         if not self.is_ready:
#             rospy.logwarn("Aguardando posições reais do robô...")
#             return

#         joint_names = msg.name
#         target_positions = msg.position
        
#         trajectory = JointTrajectory()
        
#         # --- CORREÇÃO VITAL: CARIMBO DE TEMPO ---
#         # Sincroniza o relógio para o Action Server não achar que a mensagem está atrasada
#         trajectory.header.stamp = rospy.Time.now()
#         trajectory.joint_names = joint_names
        
#         num_points = int(self.movement_duration * self.hz)
#         passo_tempo = self.movement_duration / num_points
#         t_array = np.linspace(passo_tempo, self.movement_duration, num_points)
        
#         for t in t_array:
#             point = JointTrajectoryPoint()
#             point.time_from_start = rospy.Duration(t)
            
#             positions, velocities, accelerations = [], [], []
            
#             for i, joint_name in enumerate(joint_names):
#                 # Agora, sem o nó falso, self.current_joints terá a posição real do Gazebo
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
#         rospy.loginfo("Trajetória Suavizada e Sincronizada Publicada!")

# if __name__ == '__main__':
#     try:
#         QuinticPlannerNode()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass











# #!/usr/bin/env python3
# # Arquivo: quintic_planner_node_140.py (GAZEBO)
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
#         rospy.Subscriber('/ur5/joint_states', JointState, self.current_state_callback)
        
#         self.traj_pub = rospy.Publisher('/ur5/planned_trajectory', JointTrajectory, queue_size=1)
        
#         self.current_joints = {}
#         self.is_ready = False
        
#         rospy.loginfo("Planejador Quíntuplo (Gazebo) Iniciado...")

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
        
#         # --- LÓGICA IDÊNTICA AO ROBÔ REAL (Evitando t=0.0) ---
#         num_points = int(self.movement_duration * self.hz)
#         passo_tempo = self.movement_duration / num_points
#         t_array = np.linspace(passo_tempo, self.movement_duration, num_points)
        
#         for t in t_array:
#             point = JointTrajectoryPoint()
#             point.time_from_start = rospy.Duration(t)
            
#             positions, velocities, accelerations = [], [], []
            
#             for i, joint_name in enumerate(joint_names):
#                 q0 = self.current_joints.get(joint_name, 0.0)
                
#                 # CORREÇÃO: Declarando a variável bruta antes de passar pelo filtro
#                 qf_bruto = target_positions[i] 

#                 # --- NOVO: FILTRO ANTI-GIRO (UNWINDING) NO MODO AUTÔNOMO ---
#                 # Garante que o planejador sempre escolha o trajeto mais curto
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
#         rospy.loginfo("Trajetória Suavizada Publicada (Gazebo)!")

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
#         rospy.Subscriber('/ur5/joint_states', JointState, self.current_state_callback)
        
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
