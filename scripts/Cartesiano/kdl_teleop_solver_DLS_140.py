#!/usr/bin/env python3
# Arquivo: unified_kdl_teleop_solver_DLS.py
# Arquitetura: ESTRATÉGIA MISTA (Streaming Suave + Reset pelo Menu)
# Proteções: Gesso Virtual (Elbow Up) + Anti-Singularidade (Wrist)
import rospy
import PyKDL as kdl
import tf_conversions.posemath as pm
from kdl_parser_py.urdf import treeFromUrdfModel
from urdf_parser_py.urdf import URDF
from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
import math

# ==============================================================================
# --- CONFIGURAÇÕES DO MODO DE TESTE ---
# ==============================================================================
TEST_MODE = False
SEND_TO_GAZEBO = True
INPUT_AS_QUATERNION = True

pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]
orient_euler = [3.14, 0.0, 0.0] 
quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  
# ==============================================================================

def resolver_salto_angular(angulo_alvo, angulo_atual):
    """
    Filtro Anti-Unwinding Blindado: Impede paralisia nos limites de ±360°.
    """
    limite_inferior = -6.28
    limite_superior = 6.28
    
    diferenca = angulo_alvo - angulo_atual
    menor_distancia = math.atan2(math.sin(diferenca), math.cos(diferenca))
    alvo_proposto = angulo_atual + menor_distancia
    
    while alvo_proposto > limite_superior:
        alvo_proposto -= 2 * math.pi
    while alvo_proposto < limite_inferior:
        alvo_proposto += 2 * math.pi
        
    return alvo_proposto

class KDLTeleopSolver:
    
    BASE_LINK = 'base_link'
    EE_LINK = 'tool0'
    POSE_TOPIC = 'unity/target_pose'
    COMMAND_TOPIC = '/ur5/planned_trajectory' 
    JOINT_STATES_TOPIC = '/ur5/joint_states'
    
    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    def __init__(self):
        rospy.loginfo("Iniciando KDL Teleop Solver (Estratégia Mista + Proteção de Singularidade)...")
        self.has_received_joints = False
        
        self._load_robot_model()
        self._initialize_kdl_solvers()
        self._setup_ros_communication()
        
        if not TEST_MODE:
            rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)

    def _load_robot_model(self):
        try:
            self.robot = URDF.from_parameter_server()
            success, kdl_tree_object = treeFromUrdfModel(self.robot)
            if not success:
                raise Exception("Falha ao construir a árvore KDL.")
                
            self.kdl_tree = kdl_tree_object
            self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
            self.num_joints = self.chain.getNrOfJoints()
        except Exception as e:
            rospy.logerr("Erro ao carregar modelo URDF: %s", str(e))
            raise

    def _initialize_kdl_solvers(self):
        self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
        
        # Aumentamos o Damping (Amortecimento) do solver wdls para movimentos mais suaves
        self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        self.kdl_solver_vel.setLambda(2.0) 
        
        # =================================================================
        # --- O GESSO VIRTUAL (JOINT LIMITS EXTREMOS) ---
        # =================================================================
        self.q_min = kdl.JntArray(self.num_joints)
        self.q_max = kdl.JntArray(self.num_joints)
        
        # 1. Liberdade total por padrão (±360 graus)
        for i in range(self.num_joints):
            self.q_min[i] = -2 * math.pi
            self.q_max[i] =  2 * math.pi

        # 2. Restrição do Ombro (Elbow Up Seguro)
        self.q_min[1] = -2.3 
        self.q_max[1] =  0.0
        
        # 3. Restrição do Cotovelo (Elbow Up Seguro)
        self.q_min[2] = -math.pi       
        self.q_max[2] = -0.1           
        
        # 4. BLINDAGEM DO PUNHO (Anti-Singularidade)
        # Mantém a junta wrist_2 entre ~5° e ~171°, impedindo que os eixos 1 e 3 se alinhem.
        self.q_min[4] =  0.1
        self.q_max[4] =  3.0
        # =================================================================
        
        self.kdl_solver_pos = kdl.ChainIkSolverPos_NR_JL(
            self.chain, self.q_min, self.q_max, self.kdl_solver_fk, self.kdl_solver_vel, maxiter=50, eps=1e-3
        )
        
        self.q_init = kdl.JntArray(self.num_joints)
        self.last_q_out = kdl.JntArray(self.num_joints)
        self.is_first_ik = True 

    def _setup_ros_communication(self): 
        self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
        rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
        if not TEST_MODE:
            self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
            self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
            rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

    def joint_states_callback(self, msg):
        for i, joint_name in enumerate(self.JOINT_NAMES):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.q_init[i] = msg.position[idx]
        self.has_received_joints = True

    def process_pose(self, pose_stamped_msg):
        if not self.has_received_joints:
            return False

        # =====================================================================
        # DETECTOR DE MENU (ESTRATÉGIA MISTA)
        # Se o braço for movido por outra fonte (Menu autônomo), ele reseta
        # a memória e sincroniza com a realidade para não dar "tranco".
        # =====================================================================
        if not self.is_first_ik:
            max_drift = 0.0
            for atual, memoria in zip(self.q_init, self.last_q_out):
                dif = atual - memoria
                drift = abs(math.atan2(math.sin(dif), math.cos(dif)))
                if drift > max_drift:
                    max_drift = drift
            
            # Tolerância de ~11 graus antes de considerar que o menu atuou
            if max_drift > 0.2:
                rospy.logwarn("Menu atuou! Sincronizando a memória do KDL com a nova posição física.")
                self.is_first_ik = True
        # =====================================================================

        target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        q_out = kdl.JntArray(self.num_joints)
        
        # Semente Mista: Usa a realidade no primeiro momento, e a memória para continuidade no VR
        seed = self.q_init if self.is_first_ik else self.last_q_out
        result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
        if result >= 0: 
            juntas_normalizadas = [resolver_salto_angular(q_out[i], seed[i]) for i in range(self.num_joints)]
            
            for i in range(self.num_joints):
                self.last_q_out[i] = juntas_normalizadas[i]
                
            self.is_first_ik = False

            # Painel de Log
            juntas_graus = [math.degrees(j) for j in juntas_normalizadas]
            log_msg = "\n" + "="*55 + "\n"
            log_msg += "[KDL SOLVER] Estratégia Mista - Punho Seguro\n"
            log_msg += "="*55 + "\n"
            for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
                log_msg += f" -> {nome}: {angulo:7.2f}°\n"
            log_msg += "="*55
            
            rospy.loginfo_throttle(0.5, log_msg)

            if SEND_TO_GAZEBO:
                self._publish_joint_command(juntas_normalizadas)
                
            return True

        return False

    def pose_callback(self, data):
        self.process_pose(data)

    def autonomous_pose_callback(self, pose_stamped_msg):
        if not self.has_received_joints:
            return
            
        target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        q_out = kdl.JntArray(self.num_joints)
        
        # =====================================================================
        # A CURA DO ENVENENAMENTO DE SEMENTE:
        # Movimentos autônomos discretos SEMPRE usam a posição real do robô
        # como base de cálculo. Jogamos a memória viciada do VR no lixo.
        # =====================================================================
        seed = self.q_init 
        
        result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
        if result >= 0: 
            juntas_normalizadas = [resolver_salto_angular(q_out[i], seed[i]) for i in range(self.num_joints)]
            
            # Atualiza a memória com o cálculo seguro e recém-criado
            for i in range(self.num_joints):
                self.last_q_out[i] = juntas_normalizadas[i]
                
            # FORÇA O RESET DO VR: 
            # Avisa o sistema que o próximo movimento da mão deve partir daqui
            self.is_first_ik = True
            
            js_msg = JointState()
            js_msg.name = self.JOINT_NAMES + ['finger_joint']
            pos_list = juntas_normalizadas
            pos_list.append(0.0)
            js_msg.position = pos_list
            self.quintic_pub.publish(js_msg)

    def _publish_joint_command(self, joint_positions):
        joint_positions_list = list(joint_positions)
        traj_msg = JointTrajectory()
        traj_msg.header = Header()
        traj_msg.header.stamp = rospy.Time.now()
        traj_msg.header.frame_id = self.BASE_LINK
        traj_msg.joint_names = self.JOINT_NAMES
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions_list
        point.velocities = [0.0] * self.num_joints
        point.accelerations = [0.0] * self.num_joints
        
        # Amortecedor Dinâmico (Anti-solavanco)
        max_delta = 0.0
        for alvo, atual in zip(joint_positions_list, self.q_init):
            diferenca = alvo - atual
            delta_circular = abs(math.atan2(math.sin(diferenca), math.cos(diferenca)))
            if delta_circular > max_delta:
                max_delta = delta_circular
                
        max_rad_per_sec = 2.0 
        tempo_dinamico = max(0.2, max_delta / max_rad_per_sec)
        
        if tempo_dinamico > 0.15:
            rospy.logwarn_throttle(1.0, f"[SEGURANÇA] Sincronizando Pinça. Amortecedor: {tempo_dinamico:.2f}s")

        point.time_from_start = rospy.Duration(tempo_dinamico) 
        
        traj_msg.points.append(point)
        self.command_pub.publish(traj_msg)

def main():
    rospy.init_node('kdl_teleop_solver_node', anonymous=True)
    solver = KDLTeleopSolver()
    
    if TEST_MODE:
        rospy.loginfo("--- INICIANDO ROTINA DE TESTE ---")
        test_pose = Pose()
        test_pose.position.x = pos[0]
        test_pose.position.y = pos[1]
        test_pose.position.z = pos[2]
        
        if INPUT_AS_QUATERNION:
            test_pose.orientation.x = quat_input[0]
            test_pose.orientation.y = quat_input[1]
            test_pose.orientation.z = quat_input[2]
            test_pose.orientation.w = quat_input[3]
        else:
            import tf.transformations
            quat = tf.transformations.quaternion_from_euler(orient_euler[0], orient_euler[1], orient_euler[2])
            test_pose.orientation.x, test_pose.orientation.y, test_pose.orientation.z, test_pose.orientation.w = quat
        
        def timer_callback(event):
            test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
            solver.process_pose(test_stamped)

        rospy.Timer(rospy.Duration(1.0), timer_callback)
        rospy.spin()
    else:
        rospy.spin()

if __name__ == '__main__':
    main()










# #!/usr/bin/env python3
# # Arquivo: unified_kdl_teleop_solver_DLS.py
# # Arquitetura Closed-Loop com Gesso Virtual (NR_JL) somente com o joint_state.
# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState
# import math

# # ==============================================================================
# # --- CONFIGURAÇÕES DO MODO DE TESTE ---
# # ==============================================================================
# TEST_MODE = False
# SEND_TO_GAZEBO = True
# INPUT_AS_QUATERNION = True

# pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]
# orient_euler = [3.14, 0.0, 0.0] 
# quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  
# # ==============================================================================

# def resolver_salto_angular(angulo_alvo, angulo_atual):
#     """
#     Filtro Anti-Unwinding Blindado: Impede que o robô paralise 
#     ao bater nos limites físicos de ±360°.
#     """
#     limite_inferior = -6.28
#     limite_superior = 6.28
    
#     diferenca = angulo_alvo - angulo_atual
#     menor_distancia = math.atan2(math.sin(diferenca), math.cos(diferenca))
#     alvo_proposto = angulo_atual + menor_distancia
    
#     while alvo_proposto > limite_superior:
#         alvo_proposto -= 2 * math.pi
#     while alvo_proposto < limite_inferior:
#         alvo_proposto += 2 * math.pi
        
#     return alvo_proposto

# class KDLTeleopSolver:
    
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/planned_trajectory' 
#     JOINT_STATES_TOPIC = '/ur5/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver (Closed-Loop + Gesso Virtual)...")
#         self.has_received_joints = False
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
#             if not success:
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo URDF: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
#         self.kdl_solver_vel.setLambda(1) # Damping factor
        
#         # =================================================================
#         # --- O GESSO VIRTUAL (JOINT LIMITS) ---
#         # =================================================================
#         self.q_min = kdl.JntArray(self.num_joints)
#         self.q_max = kdl.JntArray(self.num_joints)
        
#         # 1. Liberdade total por padrão (±360 graus)
#         for i in range(self.num_joints):
#             self.q_min[i] = -2 * math.pi
#             self.q_max[i] =  2 * math.pi

#         # 2. Restrição do Ombro (Elbow Up Seguro)
#         self.q_min[1] = -2.0 
#         self.q_max[1] =  0.0
        
#         # 3. Restrição do Cotovelo (Elbow Up Seguro)
#         self.q_min[2] = -math.pi       
#         self.q_max[2] = -0.1           
#         # =================================================================
        
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR_JL(
#             self.chain, self.q_min, self.q_max, self.kdl_solver_fk, self.kdl_solver_vel, maxiter=50, eps=1e-3
#         )
        
#         # Variável única e absoluta da realidade física do robô
#         self.q_init = kdl.JntArray(self.num_joints)

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
#             self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
#             rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return False

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         # =====================================================================
#         # A CORREÇÃO: Ancoragem absoluta na realidade física (Closed-Loop)
#         # O KDL agora é obrigado a olhar para a posição real antes de calcular
#         # =====================================================================
#         seed = self.q_init 
#         result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             juntas_normalizadas = [resolver_salto_angular(q_out[i], seed[i]) for i in range(self.num_joints)]
            
#             # Painel de Log
#             juntas_graus = [math.degrees(j) for j in juntas_normalizadas]
#             log_msg = "\n" + "="*55 + "\n"
#             log_msg += "[KDL SOLVER] Cinemática Closed-Loop (Punho Seguro)\n"
#             log_msg += "="*55 + "\n"
#             for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
#                 log_msg += f" -> {nome}: {angulo:7.2f}°\n"
#             log_msg += "="*55
            
#             rospy.loginfo_throttle(0.5, log_msg)

#             if SEND_TO_GAZEBO:
#                 self._publish_joint_command(juntas_normalizadas)
                
#             return True

#         return False

#     def pose_callback(self, data):
#         self.process_pose(data)

#     def autonomous_pose_callback(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return
            
#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         # Ancoragem fechada também no modo autônomo
#         seed = self.q_init 
#         result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             juntas_normalizadas = [resolver_salto_angular(q_out[i], seed[i]) for i in range(self.num_joints)]
            
#             js_msg = JointState()
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
#             pos_list = juntas_normalizadas
#             pos_list.append(0.0)
#             js_msg.position = pos_list
#             self.quintic_pub.publish(js_msg)

#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
#         traj_msg = JointTrajectory()
#         traj_msg.header = Header()
#         traj_msg.header.stamp = rospy.Time.now()
#         traj_msg.header.frame_id = self.BASE_LINK
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         point.velocities = [0.0] * self.num_joints
#         point.accelerations = [0.0] * self.num_joints
        
#         # Amortecedor Dinâmico (Anti-solavanco)
#         max_delta = 0.0
#         for alvo, atual in zip(joint_positions_list, self.q_init):
#             diferenca = alvo - atual
#             delta_circular = abs(math.atan2(math.sin(diferenca), math.cos(diferenca)))
#             if delta_circular > max_delta:
#                 max_delta = delta_circular
                
#         max_rad_per_sec = 2.0 
#         tempo_dinamico = max(0.2, max_delta / max_rad_per_sec)
        
#         if tempo_dinamico > 0.15:
#             rospy.logwarn_throttle(1.0, f"[SEGURANÇA] Sincronizando Pinça. Amortecedor: {tempo_dinamico:.2f}s")

#         point.time_from_start = rospy.Duration(tempo_dinamico) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE ---")
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         if INPUT_AS_QUATERNION:
#             test_pose.orientation.x = quat_input[0]
#             test_pose.orientation.y = quat_input[1]
#             test_pose.orientation.z = quat_input[2]
#             test_pose.orientation.w = quat_input[3]
#         else:
#             import tf.transformations
#             quat = tf.transformations.quaternion_from_euler(orient_euler[0], orient_euler[1], orient_euler[2])
#             test_pose.orientation.x, test_pose.orientation.y, test_pose.orientation.z, test_pose.orientation.w = quat
        
#         def timer_callback(event):
#             test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
#             solver.process_pose(test_stamped)

#         rospy.Timer(rospy.Duration(1.0), timer_callback)
#         rospy.spin()
#     else:
#         rospy.spin()

# if __name__ == '__main__':
#     main()













# #!/usr/bin/env python3
# # Arquivo: unified_kdl_teleop_solver_DLS.py
# # Ajustado com "Gesso Virtual" (NR_JL) para forçar postura segura (Elbow Up)Sai o ChainIkSolverPos_NR (livre) e entra o ChainIkSolverPos_NR_JL (Newton-Raphson com Joint Limits).
# # 
# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState
# import math

# # ==============================================================================
# # --- CONFIGURAÇÕES DO MODO DE TESTE ---
# # ==============================================================================
# TEST_MODE = False
# SEND_TO_GAZEBO = True
# INPUT_AS_QUATERNION = True

# pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]
# orient_euler = [3.14, 0.0, 0.0] 
# quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  
# # ==============================================================================

# def resolver_salto_angular(angulo_alvo, angulo_atual):
#     """
#     Filtro Anti-Unwinding Blindado: Resolve o caminho mais curto e 
#     impede que o robô paralise ao bater nos limites físicos de ±360°.
#     """
#     limite_inferior = -6.28
#     limite_superior = 6.28
    
#     diferenca = angulo_alvo - angulo_atual
#     menor_distancia = math.atan2(math.sin(diferenca), math.cos(diferenca))
#     alvo_proposto = angulo_atual + menor_distancia
    
#     while alvo_proposto > limite_superior:
#         alvo_proposto -= 2 * math.pi
#     while alvo_proposto < limite_inferior:
#         alvo_proposto += 2 * math.pi
        
#     return alvo_proposto

# class KDLTeleopSolver:
    
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/planned_trajectory' 
#     JOINT_STATES_TOPIC = '/ur5/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver Unificado com Gesso Virtual...")
#         self.has_received_joints = False
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
#             if not success:
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo URDF: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
#         self.kdl_solver_vel.setLambda(1) # Damping factor
        
#         # =================================================================
#         # --- O GESSO VIRTUAL (JOINT LIMITS) ---
#         # =================================================================
#         self.q_min = kdl.JntArray(self.num_joints)
#         self.q_max = kdl.JntArray(self.num_joints)
        
#         # 1. Liberdade total por padrão (±360 graus)
#         for i in range(self.num_joints):
#             self.q_min[i] = -2 * math.pi
#             self.q_max[i] =  2 * math.pi

#         # 2. Restrição do Ombro (shoulder_lift_joint = índice 1)
#         # O seu log perigoso mostrou o ombro descendo para -139°. 
#         # Restringimos para não passar de -114° (~ -2.0 rad), forçando o robô a buscar o objeto dobrando o cotovelo.
#         self.q_min[1] = -2.0 
#         self.q_max[1] =  0.0
        
#         # 3. Restrição do Cotovelo (elbow_joint = índice 2)
#         # Mantemos sempre do lado negativo para garantir o "Elbow Up".
#         self.q_min[2] = -math.pi       # Dobra máxima (-180°)
#         self.q_max[2] = -0.1           # Limite antes de ficar perigosamente reto (-5.7°)
#         # =================================================================
        
#         # Trocamos o solver NR pelo NR_JL (Newton-Raphson com Joint Limits)
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR_JL(
#             self.chain, self.q_min, self.q_max, self.kdl_solver_fk, self.kdl_solver_vel, maxiter=50, eps=1e-3
#         )
        
#         self.q_init = kdl.JntArray(self.num_joints)
#         self.last_q_out = kdl.JntArray(self.num_joints)
#         self.is_first_ik = True 

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
#             self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
#             rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return False

#         if not self.is_first_ik:
#             max_drift = 0.0
#             for atual, memoria in zip(self.q_init, self.last_q_out):
#                 dif = atual - memoria
#                 drift = abs(math.atan2(math.sin(dif), math.cos(dif)))
#                 if drift > max_drift:
#                     max_drift = drift
            
#             if max_drift > 0.2:
#                 rospy.logwarn("Menu atuou! Sincronizando a memória do KDL com a nova posição.")
#                 self.is_first_ik = True

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         seed = self.q_init if self.is_first_ik else self.last_q_out
#         result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             juntas_normalizadas = [resolver_salto_angular(q_out[i], seed[i]) for i in range(self.num_joints)]
            
#             for i in range(self.num_joints):
#                 self.last_q_out[i] = juntas_normalizadas[i]
                
#             self.is_first_ik = False

#             juntas_graus = [math.degrees(j) for j in juntas_normalizadas]
            
#             log_msg = "\n" + "="*55 + "\n"
#             log_msg += "[KDL SOLVER] Cinemática Inversa Segura (Com Limites)\n"
#             log_msg += "="*55 + "\n"
#             for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
#                 log_msg += f" -> {nome}: {angulo:7.2f}°\n"
#             log_msg += "="*55
            
#             rospy.loginfo_throttle(0.5, log_msg)

#             if SEND_TO_GAZEBO:
#                 self._publish_joint_command(juntas_normalizadas)
                
#             return True

#         return False

#     def pose_callback(self, data):
#         self.process_pose(data)

#     def autonomous_pose_callback(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return
            
#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
#         seed = self.q_init if self.is_first_ik else self.last_q_out
#         result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             juntas_normalizadas = [resolver_salto_angular(q_out[i], seed[i]) for i in range(self.num_joints)]
            
#             for i in range(self.num_joints):
#                 self.last_q_out[i] = juntas_normalizadas[i]
                
#             self.is_first_ik = False
            
#             js_msg = JointState()
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
#             pos_list = juntas_normalizadas
#             pos_list.append(0.0)
#             js_msg.position = pos_list
#             self.quintic_pub.publish(js_msg)

#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
#         traj_msg = JointTrajectory()
#         traj_msg.header = Header()
#         traj_msg.header.stamp = rospy.Time.now()
#         traj_msg.header.frame_id = self.BASE_LINK
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         point.velocities = [0.0] * self.num_joints
#         point.accelerations = [0.0] * self.num_joints
        
#         max_delta = 0.0
#         for alvo, atual in zip(joint_positions_list, self.q_init):
#             diferenca = alvo - atual
#             delta_circular = abs(math.atan2(math.sin(diferenca), math.cos(diferenca)))
#             if delta_circular > max_delta:
#                 max_delta = delta_circular
                
#         max_rad_per_sec = 2.0 
#         tempo_dinamico = max(0.2, max_delta / max_rad_per_sec)
        
#         if tempo_dinamico > 0.15:
#             rospy.logwarn_throttle(1.0, f"[SEGURANÇA] Sincronizando Pinça. Amortecedor: {tempo_dinamico:.2f}s")

#         point.time_from_start = rospy.Duration(tempo_dinamico) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE ---")
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         if INPUT_AS_QUATERNION:
#             test_pose.orientation.x = quat_input[0]
#             test_pose.orientation.y = quat_input[1]
#             test_pose.orientation.z = quat_input[2]
#             test_pose.orientation.w = quat_input[3]
#         else:
#             import tf.transformations
#             quat = tf.transformations.quaternion_from_euler(orient_euler[0], orient_euler[1], orient_euler[2])
#             test_pose.orientation.x, test_pose.orientation.y, test_pose.orientation.z, test_pose.orientation.w = quat
        
#         def timer_callback(event):
#             test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
#             solver.process_pose(test_stamped)

#         rospy.Timer(rospy.Duration(1.0), timer_callback)
#         rospy.spin()
#     else:
#         rospy.spin()

# if __name__ == '__main__':
#     main()

















# #!/usr/bin/env python3
# # Arquivo: unified_kdl_teleop_solver_DLS.py
# # ajustado o script do Gazebo para espelhar as regras de segurança do mundo real
# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState
# import math

# # ==============================================================================
# # --- CONFIGURAÇÕES DO MODO DE TESTE ---
# # ==============================================================================
# TEST_MODE = False           # Ativa a rotina de teste manual
# SEND_TO_GAZEBO = True       # True: Envia comandos para o hardware/simulador

# # Escolha do modo de orientação do Alvo:
# INPUT_AS_QUATERNION = True  # True: Usa Quatérnio (x,y,z,w) | False: Usa Euler (r,p,y)

# # Posição XYZ do Alvo
# pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]
# orient_euler = [3.14, 0.0, 0.0] 
# quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  
# # ==============================================================================

# def resolver_salto_angular(angulo_alvo, angulo_atual):
#     """
#     Filtro Anti-Unwinding (Caminho Mais Curto).
#     Calcula a menor distância angular entre a pose atual e o alvo,
#     permitindo que as juntas ultrapassem a barreira dos 180 graus sem dar o giro de 360 graus.
#     """
#     diferenca = angulo_alvo - angulo_atual
#     # O atan2 descobre se é mais rápido ir pela direita ou pela esquerda
#     menor_distancia = math.atan2(math.sin(diferenca), math.cos(diferenca))
    
#     return angulo_atual + menor_distancia

# class KDLTeleopSolver:
    
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
    
#     # [REGRA DO MUNDO REAL 1]: Roteamento para o Commander Node (O Tradutor)
#     COMMAND_TOPIC = '/ur5/planned_trajectory' 
    
#     JOINT_STATES_TOPIC = '/ur5/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver Unificado (Simulação e Real)...")
#         self.has_received_joints = False
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
#             if not success:
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo URDF: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
#         self.kdl_solver_vel.setLambda(1) # Damping factor
        
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, self.kdl_solver_fk, self.kdl_solver_vel, maxiter=50, eps=1e-3
#         )
        
#         self.q_init = kdl.JntArray(self.num_joints)
#         self.last_q_out = kdl.JntArray(self.num_joints)
#         self.is_first_ik = True 

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
#             self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
#             rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return False

#         # =====================================================================
#         # --- NOVO: SINCRONIZADOR DE REALIDADE (DETECTOR DE MENU) ---
#         # Se a posição física do robô divergir da memória do KDL, 
#         # significa que o Menu (Unity) atuou. Devemos resetar a semente!
#         if not self.is_first_ik:
#             max_drift = 0.0
#             for atual, memoria in zip(self.q_init, self.last_q_out):
#                 dif = atual - memoria
#                 drift = abs(math.atan2(math.sin(dif), math.cos(dif)))
#                 if drift > max_drift:
#                     max_drift = drift
            
#             # Se o braço físico andou mais de 11 graus (~0.2 rad) sem o KDL mandar:
#             if max_drift > 0.2:
#                 rospy.logwarn("Menu atuou! Sincronizando a memória do KDL com a nova posição.")
#                 self.is_first_ik = True
#         # =====================================================================

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         seed = self.q_init if self.is_first_ik else self.last_q_out
#         result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             juntas_normalizadas = [resolver_salto_angular(q_out[i], seed[i]) for i in range(self.num_joints)]
            
#             # Garante que a memória receba apenas os ângulos limpos
#             for i in range(self.num_joints):
#                 self.last_q_out[i] = juntas_normalizadas[i]
                
#             self.is_first_ik = False

#             # ==========================================================
#             # --- PAINEL DE MONITORAMENTO DO KDL SOLVER ---
#             # ==========================================================
#             # Converte de radianos para graus para leitura humana
#             juntas_graus = [math.degrees(j) for j in juntas_normalizadas]
            
#             # Monta o painel de texto
#             log_msg = "\n" + "="*55 + "\n"
#             log_msg += "[KDL SOLVER] Cinemática Inversa (Shortest Path)\n"
#             log_msg += "="*55 + "\n"
#             for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
#                 log_msg += f" -> {nome}: {angulo:7.2f}°\n"
#             log_msg += "="*55
            
#             # Imprime no terminal a cada 0.5 segundos (evita spam/lag)
#             rospy.loginfo_throttle(0.5, log_msg)
#             # ==========================================================

#             if SEND_TO_GAZEBO:
#                 self._publish_joint_command(juntas_normalizadas)
                
#             return True # O retorno só acontece no final do bloco de sucesso

#         return False
    

#     def pose_callback(self, data):
#         self.process_pose(data)

#     def autonomous_pose_callback(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return
            
#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
#         seed = self.q_init if self.is_first_ik else self.last_q_out
#         result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             juntas_normalizadas = [resolver_salto_angular(q_out[i], seed[i]) for i in range(self.num_joints)]
            
#             # --- CORREÇÃO DA AMNÉSIA NO MODO AUTÔNOMO ---
#             for i in range(self.num_joints):
#                 self.last_q_out[i] = juntas_normalizadas[i]
                
#             self.is_first_ik = False
            
#             js_msg = JointState()
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
#             pos_list = juntas_normalizadas
#             pos_list.append(0.0)
#             js_msg.position = pos_list
#             self.quintic_pub.publish(js_msg)

#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
#         traj_msg = JointTrajectory()
#         traj_msg.header = Header()
#         traj_msg.header.stamp = rospy.Time.now()
#         traj_msg.header.frame_id = self.BASE_LINK
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         point.velocities = [0.0] * self.num_joints
#         point.accelerations = [0.0] * self.num_joints
        
#         # =================================================================
#         # --- NOVO: AMORTECEDOR DINÂMICO DE REENGAJAMENTO (ANTI-SOLAVANCO) ---
#         # =================================================================
#         max_delta = 0.0
#         for alvo, atual in zip(joint_positions_list, self.q_init):
#             # A matemática do mundo redondo: 179 e -179 estão a 2 graus de distância, não 358!
#             diferenca = alvo - atual
#             delta_circular = abs(math.atan2(math.sin(diferenca), math.cos(diferenca)))
            
#             if delta_circular > max_delta:
#                 max_delta = delta_circular
                
#         # Define uma velocidade máxima ágil para reengajamento (2.0 rad/s = ~114 graus/s)
#         max_rad_per_sec = 2.0 
        
#         # Calcula o tempo baseado na distância real
#         tempo_dinamico = max(0.2, max_delta / max_rad_per_sec)
        
#         if tempo_dinamico > 0.15:
#             rospy.logwarn_throttle(1.0, f"[SEGURANÇA] Sincronizando Pinça. Amortecedor: {tempo_dinamico:.2f}s")

#         point.time_from_start = rospy.Duration(tempo_dinamico) 
#         # =================================================================
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE ---")
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         if INPUT_AS_QUATERNION:
#             test_pose.orientation.x = quat_input[0]
#             test_pose.orientation.y = quat_input[1]
#             test_pose.orientation.z = quat_input[2]
#             test_pose.orientation.w = quat_input[3]
#         else:
#             import tf.transformations
#             quat = tf.transformations.quaternion_from_euler(orient_euler[0], orient_euler[1], orient_euler[2])
#             test_pose.orientation.x, test_pose.orientation.y, test_pose.orientation.z, test_pose.orientation.w = quat
        
#         def timer_callback(event):
#             test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
#             solver.process_pose(test_stamped)

#         rospy.Timer(rospy.Duration(1.0), timer_callback)
#         rospy.spin()
#     else:
#         rospy.spin()

# if __name__ == '__main__':
#     main()










# #!/usr/bin/env python3
# # Arquivo: gazebo_kdl_teleop_solver_DLS.py
# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState
# import math

# # ==============================================================================
# # --- CONFIGURAÇÕES DO MODO DE TESTE ---
# # ==============================================================================
# TEST_MODE = False             # Ativa a rotina de teste manual
# SEND_TO_GAZEBO = True       # True: Envia comandos para o Gazebo | False: Apenas terminal

# # Escolha do modo de orientação do Alvo:
# INPUT_AS_QUATERNION = True # True: Usa Quatérnio (x,y,z,w) | False: Usa Euler (r,p,y)

# # Posição XYZ do Alvo (Comum para ambos os modos)
# pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]

# # Opção A: Orientação em Euler (Usada se INPUT_AS_QUATERNION = False)
# orient_euler = [3.14, 0.0, 0.0] # Roll, Pitch, Yaw (Garra apontada para baixo)

# # Opção B: Orientação em Quatérnio (Usada se INPUT_AS_QUATERNION = True)
# quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  # Estrutura: [x, y, z, w]
# # ==============================================================================

# def normalizar_angulo(angulo):
#     """
#     Filtro Anti-Tornado: Força qualquer ângulo a ficar estritamente
#     entre -π e +π (-180° a +180°), evitando giros mortais.
#     """
#     return math.atan2(math.sin(angulo), math.cos(angulo))

# class KDLTeleopSolver:
#     """
#     Servidor de cinemática inversa usando KDL para controle do UR5 no Gazebo.
#     Implementa Damped Least Squares (DLS) e blindagem trigonométrica.
#     """
    
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
    
#     # [AJUSTE GAZEBO 1]: O Gazebo geralmente usa o controlador sem a tag "scaled"
#     COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command' 
    
#     # [AJUSTE GAZEBO 2]: Lembra que no seu Gazebo o tópico tinha o prefixo /ur5?
#     JOINT_STATES_TOPIC = '/ur5/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 
#         'shoulder_lift_joint', 
#         'elbow_joint',
#         'wrist_1_joint', 
#         'wrist_2_joint', 
#         'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver com amortecimento DLS (GAZEBO)...")
        
#         self.has_received_joints = False
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
#         else:
#             rospy.loginfo("Modo ATIVO: Teste Manual Ativo (SEND_TO_GAZEBO = %s)", str(SEND_TO_GAZEBO))

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
#             if not success:
#                 rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#             rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo do robô do Parameter Server: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
#         damping_factor = 1   # Fator de amortecimento para o DLS
#         self.kdl_solver_vel.setLambda(damping_factor)
        
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, 
#             self.kdl_solver_fk, 
#             self.kdl_solver_vel,
#             maxiter=50,   
#             eps=1e-3   
#         )
        
#         self.q_init = kdl.JntArray(self.num_joints)
        
#         # --- NOVAS VARIÁVEIS DE MEMÓRIA MUSCULAR ---
#         self.last_q_out = kdl.JntArray(self.num_joints)
#         self.is_first_ik = True 

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
#             self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
#             rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             rospy.logwarn_throttle(2.0, "KDL: Aguardando semente do Gazebo...")
#             return False

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         # Usa o Gazebo no 1º frame. Depois, usa a matemática anterior para evitar o FLIP.
#         seed = self.q_init if self.is_first_ik else self.last_q_out
        
#         result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             self.last_q_out = q_out      
#             self.is_first_ik = False     

#             # --- BLINDAGEM MATEMÁTICA ---
#             juntas_normalizadas = [normalizar_angulo(q_out[i]) for i in range(self.num_joints)]
#             juntas_graus = [math.degrees(j) for j in juntas_normalizadas]
            
#             print("\n" + "="*60)
#             print("[KDL SOLVER] Solução de Cinemática Inversa blindada e congelada!")
#             print("="*60)
#             for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
#                 print(f" -> {nome}: {angulo:.2f}°")
#             print("="*60)

#             if SEND_TO_GAZEBO:
#                 self._publish_joint_command(juntas_normalizadas)
#                 print("[INFO] Comando enviado com segurança ao Gazebo.")
#             else:
#                 print("[INFO] Modo Visualização Ativo: Comando NÃO enviado ao Gazebo.")
                
#             return True
#         else:
#             rospy.logwarn_throttle(1, "Alvo fora de alcance ou singularidade extrema. Código: %d", result)
#             return False

#     def pose_callback(self, data):
#         self.process_pose(data)

#     def autonomous_pose_callback(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         seed = self.q_init if self.is_first_ik else self.last_q_out
#         result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             self.last_q_out = q_out
#             self.is_first_ik = False

#             juntas_normalizadas = [normalizar_angulo(q_out[i]) for i in range(self.num_joints)]
            
#             js_msg = JointState()
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
#             pos_list = juntas_normalizadas
#             pos_list.append(0.0)
#             js_msg.position = pos_list
#             self.quintic_pub.publish(js_msg)

#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
        
#         # [AJUSTE GAZEBO 3]: Para simulação ser fluida/responsiva, retornamos ao tempo rápido.
#         # No robô real continuará 1.0 para segurança.
#         point.time_from_start = rospy.Duration(0.04) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO GAZEBO ---")
        
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         if INPUT_AS_QUATERNION:
#             rospy.loginfo("[TEST_MODE] Utilizando orientação via QUATÉRNIO direto.")
#             test_pose.orientation.x = quat_input[0]
#             test_pose.orientation.y = quat_input[1]
#             test_pose.orientation.z = quat_input[2]
#             test_pose.orientation.w = quat_input[3]
#         else:
#             rospy.loginfo("[TEST_MODE] Utilizando orientação via ÂNGULOS DE EULER.")
#             import tf.transformations
#             quat = tf.transformations.quaternion_from_euler(orient_euler[0], orient_euler[1], orient_euler[2])
#             test_pose.orientation.x = quat[0]
#             test_pose.orientation.y = quat[1]
#             test_pose.orientation.z = quat[2]
#             test_pose.orientation.w = quat[3]
        
#         def timer_callback(event):
#             test_stamped = PoseStamped(
#                 header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), 
#                 pose=test_pose
#             )
#             solver.process_pose(test_stamped)

#         rospy.Timer(rospy.Duration(1.0), timer_callback)
#         rospy.spin()
            
#     else:
#         rospy.spin()

# if __name__ == '__main__':
#     main()













# #!/usr/bin/env python3
# # Esta é uma cópia da versão anterior, pulando 1
# # Arquivo: kdl_teleop_solver_DLS_140.py
# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState
# import math

# # ==============================================================================
# # --- CONFIGURAÇÕES DO MODO DE TESTE ---
# # ==============================================================================
# TEST_MODE = False           # Ativa a rotina de teste manual
# SEND_TO_GAZEBO = True      # True: Envia comandos para o Gazebo | False: Apenas visualiza no terminal

# # Escolha do modo de orientação do Alvo:
# INPUT_AS_QUATERNION = True # True: Usa Quatérnio (x,y,z,w) | False: Usa Euler (r,p,y)

# # Posição XYZ do Alvo (Comum para ambos os modos)
# pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]

# # Opção A: Orientação em Euler (Usada se INPUT_AS_QUATERNION = False)
# orient_euler = [3.14, 0.0, 0.0] # Roll, Pitch, Yaw (Garra apontada para baixo)

# # Opção B: Orientação em Quatérnio (Usada se INPUT_AS_QUATERNION = True)
# # Exemplo abaixo: Equivale a [3.14, 0.0, 0.0] em Euler (Olhando para baixo no UR5)
# quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  # Estrutura: [x, y, z, w]
# # ==============================================================================

# class KDLTeleopSolver:
#     """
#     Servidor de cinemática inversa usando KDL para controle do UR5.
#     Implementa Damped Least Squares (DLS) para estabilidade em singularidades.
#     """
    
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
#     JOINT_STATES_TOPIC = '/ur5/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver com amortecimento DLS...")
        
#         self.has_received_joints = False
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
#         else:
#             rospy.loginfo("Modo ATIVO: Teste Manual Ativo (SEND_TO_GAZEBO = %s)", str(SEND_TO_GAZEBO))

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
#             if not success:
#                 rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#             rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo do robô do Parameter Server: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
#         damping_factor = 1   # CALIBRAÇÃO --> Fator de amortecimento para o DLS (Damped Least Squares)
#         self.kdl_solver_vel.setLambda(damping_factor)
        
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, 
#             self.kdl_solver_fk, 
#             self.kdl_solver_vel,
#             maxiter=50,   
#             eps=1e-3   
#         )
        
#         self.q_init = kdl.JntArray(self.num_joints)

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
#             self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
#             rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             rospy.logwarn_throttle(2.0, "KDL: Aguardando semente física do Gazebo...")
#             return False

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             juntas_graus = [math.degrees(q_out[i]) for i in range(self.num_joints)]
            
#             print("\n" + "="*60)
#             print("[KDL SOLVER] Solução de Cinemática Inversa encontrada!")
#             print("="*60)
#             for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
#                 print(f" -> {nome}: {angulo:.2f}°")
#             print("="*60)

#             if SEND_TO_GAZEBO:
#                 self._publish_joint_command(q_out)
#                 print("[INFO] Comando enviado com sucesso ao Gazebo.")
#             else:
#                 print("[INFO] Modo Visualização Ativo: Comando NÃO enviado ao Gazebo.")
                
#             return True
#         else:
#             rospy.logwarn_throttle(1, "Alvo fora de alcance ou singularidade extrema. Código: %d", result)
#             return False

#     def pose_callback(self, data):
#         self.process_pose(data)

#     def autonomous_pose_callback(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             js_msg = JointState()
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
#             pos_list = list(q_out)
#             pos_list.append(0.0) 
#             js_msg.position = pos_list
#             self.quintic_pub.publish(js_msg)

#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         point.time_from_start = rospy.Duration(0.04)  # PARÂMETRO para calibrar o Tempo para alcançar o ponto
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO GAZEBO ---")
        
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         # Lógica de seleção do tipo de entrada de orientação
#         if INPUT_AS_QUATERNION:
#             rospy.loginfo("[TEST_MODE] Utilizando orientação via QUATÉRNIO direto.")
#             test_pose.orientation.x = quat_input[0]
#             test_pose.orientation.y = quat_input[1]
#             test_pose.orientation.z = quat_input[2]
#             test_pose.orientation.w = quat_input[3]
#         else:
#             rospy.loginfo("[TEST_MODE] Utilizando orientação via ÂNGULOS DE EULER.")
#             import tf.transformations
#             quat = tf.transformations.quaternion_from_euler(orient_euler[0], orient_euler[1], orient_euler[2])
#             test_pose.orientation.x = quat[0]
#             test_pose.orientation.y = quat[1]
#             test_pose.orientation.z = quat[2]
#             test_pose.orientation.w = quat[3]
        
#         def timer_callback(event):
#             test_stamped = PoseStamped(
#                 header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), 
#                 pose=test_pose
#             )
#             solver.process_pose(test_stamped)

#         rospy.Timer(rospy.Duration(1.0), timer_callback)
#         rospy.spin()
            
#     else:
#         rospy.spin()

# if __name__ == '__main__':
#     main()




















# #!/usr/bin/env python3
# # alteração em 02/08/2026
# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState
# import math

# # ==============================================================================
# # --- CONFIGURAÇÕES DO MODO DE TESTE ---
# # ==============================================================================
# TEST_MODE = False
# SEND_TO_GAZEBO = True
# INPUT_AS_QUATERNION = True
# pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]
# orient_euler = [3.14, 0.0, 0.0] 
# quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  
# # ==============================================================================

# class KDLTeleopSolver:
    
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
#     JOINT_STATES_TOPIC = '/ur5/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver com Interpolador Temporal...")
        
#         self.has_received_joints = False
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
#         else:
#             rospy.loginfo("Modo ATIVO: Teste Manual Ativo")

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
#             if not success:
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
            
#         except Exception as e:
#             rospy.logerr("Erro URDF: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
#         self.kdl_solver_vel.setLambda(0.1) 
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, self.kdl_solver_fk, self.kdl_solver_vel, maxiter=50, eps=1e-3
#         )
#         self.q_init = kdl.JntArray(self.num_joints)

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
#             self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
#             rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         if not self.has_received_joints: return False

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             if SEND_TO_GAZEBO:
#                 # Extração nativa para evitar quebras de PyKDL no Python 3
#                 pos_list = [q_out[i] for i in range(self.num_joints)]
#                 self._publish_joint_command(pos_list)
#             return True
#         return False

#     def pose_callback(self, data):
#         self.process_pose(data)

#     def autonomous_pose_callback(self, pose_stamped_msg):
#         if not self.has_received_joints: return

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             js_msg = JointState()
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
#             pos_list = [q_out[i] for i in range(self.num_joints)]
#             pos_list.append(0.0) 
#             js_msg.position = pos_list
#             self.quintic_pub.publish(js_msg)

#     def _publish_joint_command(self, joint_positions):
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions
        
#         # ==============================================================
#         # O SEGREDO DA FLUIDEZ NO GAZEBO: BUFFER TEMPORAL
#         # O controlador construirá a Spline assumindo que tem 80ms para chegar.
#         # Como o Unity enviará novas coordenadas a cada ~30ms, o alvo é
#         # constantemente atualizado ANTES dos motores precisarem frear.
#         # Se o braço ainda estiver vibrando um pouco, aumente o rospy.Duration(0.08) para 0.10 ou 0.12.
#         # ==============================================================
#         point.time_from_start = rospy.Duration(0.08) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
#     rospy.spin()

# if __name__ == '__main__':
#     main()
















# #!/usr/bin/env python3

# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState
# import math

# # ==============================================================================
# # --- CONFIGURAÇÕES DO MODO DE TESTE ---
# # ==============================================================================
# TEST_MODE = False           # Ativa a rotina de teste manual
# SEND_TO_GAZEBO = True      # True: Envia comandos para o Gazebo | False: Apenas visualiza no terminal

# # Escolha do modo de orientação do Alvo:
# INPUT_AS_QUATERNION = True # True: Usa Quatérnio (x,y,z,w) | False: Usa Euler (r,p,y)

# # Posição XYZ do Alvo (Comum para ambos os modos)
# pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]

# # Opção A: Orientação em Euler (Usada se INPUT_AS_QUATERNION = False)
# orient_euler = [3.14, 0.0, 0.0] # Roll, Pitch, Yaw (Garra apontada para baixo)

# # Opção B: Orientação em Quatérnio (Usada se INPUT_AS_QUATERNION = True)
# # Exemplo abaixo: Equivale a [3.14, 0.0, 0.0] em Euler (Olhando para baixo no UR5)
# quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  # Estrutura: [x, y, z, w]
# # ==============================================================================

# class KDLTeleopSolver:
#     """
#     Servidor de cinemática inversa usando KDL para controle do UR5.
#     Implementa Damped Least Squares (DLS) para estabilidade em singularidades.
#     """
    
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
#     JOINT_STATES_TOPIC = '/ur5/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver com amortecimento DLS...")
        
#         self.has_received_joints = False
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
#         else:
#             rospy.loginfo("Modo ATIVO: Teste Manual Ativo (SEND_TO_GAZEBO = %s)", str(SEND_TO_GAZEBO))

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
#             if not success:
#                 rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#             rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo do robô do Parameter Server: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
#         damping_factor = 0.1 
#         self.kdl_solver_vel.setLambda(damping_factor)
        
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, 
#             self.kdl_solver_fk, 
#             self.kdl_solver_vel,
#             maxiter=50,   
#             eps=1e-3   
#         )
        
#         self.q_init = kdl.JntArray(self.num_joints)

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
#             self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
#             rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             rospy.logwarn_throttle(2.0, "KDL: Aguardando semente física do Gazebo...")
#             return False

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             juntas_graus = [math.degrees(q_out[i]) for i in range(self.num_joints)]
            
#             print("\n" + "="*60)
#             print("[KDL SOLVER] Solução de Cinemática Inversa encontrada!")
#             print("="*60)
#             for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
#                 print(f" -> {nome}: {angulo:.2f}°")
#             print("="*60)

#             if SEND_TO_GAZEBO:
#                 self._publish_joint_command(q_out)
#                 print("[INFO] Comando enviado com sucesso ao Gazebo.")
#             else:
#                 print("[INFO] Modo Visualização Ativo: Comando NÃO enviado ao Gazebo.")
                
#             return True
#         else:
#             rospy.logwarn_throttle(1, "Alvo fora de alcance ou singularidade extrema. Código: %d", result)
#             return False

#     def pose_callback(self, data):
#         self.process_pose(data)

#     def autonomous_pose_callback(self, pose_stamped_msg):
#         if not self.has_received_joints:
#             return

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             js_msg = JointState()
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
#             pos_list = list(q_out)
#             pos_list.append(0.0) 
#             js_msg.position = pos_list
#             self.quintic_pub.publish(js_msg)

#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         point.time_from_start = rospy.Duration(0.01) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO GAZEBO ---")
        
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         # Lógica de seleção do tipo de entrada de orientação
#         if INPUT_AS_QUATERNION:
#             rospy.loginfo("[TEST_MODE] Utilizando orientação via QUATÉRNIO direto.")
#             test_pose.orientation.x = quat_input[0]
#             test_pose.orientation.y = quat_input[1]
#             test_pose.orientation.z = quat_input[2]
#             test_pose.orientation.w = quat_input[3]
#         else:
#             rospy.loginfo("[TEST_MODE] Utilizando orientação via ÂNGULOS DE EULER.")
#             import tf.transformations
#             quat = tf.transformations.quaternion_from_euler(orient_euler[0], orient_euler[1], orient_euler[2])
#             test_pose.orientation.x = quat[0]
#             test_pose.orientation.y = quat[1]
#             test_pose.orientation.z = quat[2]
#             test_pose.orientation.w = quat[3]
        
#         def timer_callback(event):
#             test_stamped = PoseStamped(
#                 header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), 
#                 pose=test_pose
#             )
#             solver.process_pose(test_stamped)

#         rospy.Timer(rospy.Duration(1.0), timer_callback)
#         rospy.spin()
            
#     else:
#         rospy.spin()

# if __name__ == '__main__':
#     main()











# #!/usr/bin/env python3

# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState

# # --- VARIÁVEL GLOBAL PARA MODO DE TESTE ---
# TEST_MODE = False
# # ----------------------------------------

# class KDLTeleopSolver:
#     """
#     Servidor de cinemática inversa usando KDL para controle do UR5.
#     Implementa Damped Least Squares (DLS) para estabilidade em singularidades.
#     """
    
#     # Parâmetros de Configuração
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
#     JOINT_STATES_TOPIC = '/ur5/joint_states' # Tópico para ouvir a realidade do Gazebo
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver com amortecimento DLS...")
        
#         self.has_received_joints = False # Trava de segurança
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
#         else:
#             rospy.loginfo("Modo ATIVO: Teste Manual Gazebo. Publicando pose fixa.")

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
#             if not success:
#                 rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#             rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo do robô do Parameter Server: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
#         damping_factor = 0.1 
#         self.kdl_solver_vel.setLambda(damping_factor)
        
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, 
#             self.kdl_solver_fk, 
#             self.kdl_solver_vel,
#             maxiter=200,   #maxiter=500
#             eps=1e-5   #eps=1e-3
#         )
        
#         self.q_init = kdl.JntArray(self.num_joints)

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
            
#             # ========================================================
#             # NOVA ROTA: Escuta poses autônomas e publica para o Planejador
#             # ========================================================
#             self.quintic_pub = rospy.Publisher('/unity/target_joints', JointState, queue_size=1)
#             rospy.Subscriber('/unity/target_pose_autonomous', PoseStamped, self.autonomous_pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         """Mantém a semente (q_init) perfeitamente alinhada com a realidade física."""
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 # Atualiza a semente em tempo real
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         # Trava: Não tenta calcular se ainda não sabe onde o robô está
#         if not self.has_received_joints:
#             rospy.logwarn_throttle(2.0, "KDL: Aguardando semente física do Gazebo...")
#             return False

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        
#         q_out = kdl.JntArray(self.num_joints)
        
#         # A MÁGICA: O q_init aqui agora é a foto instantânea de onde o robô parou após o botão "Pick"!
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             self._publish_joint_command(q_out)
#             # A linha "self.q_init = q_out" foi removida daqui de propósito!
#             return True
#         else:
#             rospy.logwarn_throttle(1, "Alvo fora de alcance ou singularidade extrema. Código: %d", result)
#             return False

#     def pose_callback(self, data):
#         self.process_pose(data)
#         rospy.loginfo("Pose do Unity processada: %s", data.pose)


#     def autonomous_pose_callback(self, pose_stamped_msg):
#         """ Resolve a Cinemática da Hover Pose e envia para o Planejador Quíntuplo suavizar """
#         if not self.has_received_joints:
#             return

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             js_msg = JointState()
            
#             # ========================================================
#             # O DISFARCE: Imitando exatamente o pacote do Canvas Menu
#             # Adicionamos o 'finger_joint' no final para o planejador não quebrar
#             # ========================================================
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
            
#             # Convertendo os 6 ângulos calculados e adicionando 0.0 para a garra (aberta)
#             pos_list = list(q_out)
#             pos_list.append(0.0) 
#             js_msg.position = pos_list
            
#             self.quintic_pub.publish(js_msg)
#             rospy.loginfo("Visão Ativa resolvida! Trajetória (7 juntas) enviada ao Planejador Quíntuplo.")
#         else:
#             rospy.logwarn("Visão Ativa falhou: Alvo fora de alcance.")


#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
        
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         point.time_from_start = rospy.Duration(0.01) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO GAZEBO ---")
        
#         pos = [0.3, 0.2, 0.5]
#         orient = [0.0, 1.57, 0.0] 
        
#         import tf.transformations
#         quat = tf.transformations.quaternion_from_euler(orient[0], orient[1], orient[2])
        
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         test_pose.orientation.x = quat[0]
#         test_pose.orientation.y = quat[1]
#         test_pose.orientation.z = quat[2]
#         test_pose.orientation.w = quat[3]
        
#         rate = rospy.Rate(1.0)
#         while not rospy.is_shutdown():
#             test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
#             solver.process_pose(test_stamped)
#             rate.sleep()
            
#     else:
#         rospy.spin()

# if __name__ == '__main__':
#     main()


















# #!/usr/bin/env python3

# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState

# # --- VARIÁVEL GLOBAL PARA MODO DE TESTE ---
# TEST_MODE = False
# # ----------------------------------------

# class KDLTeleopSolver:
#     """
#     Servidor de cinemática inversa usando KDL para controle do UR5.
#     Implementa Damped Least Squares (DLS) para estabilidade em singularidades.
#     """
    
#     # Parâmetros de Configuração
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
#     JOINT_STATES_TOPIC = '/ur5/joint_states' # Tópico para ouvir a realidade do Gazebo
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         rospy.loginfo("Iniciando KDL Teleop Solver com amortecimento DLS...")
        
#         self.has_received_joints = False # Trava de segurança
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
#         else:
#             rospy.loginfo("Modo ATIVO: Teste Manual Gazebo. Publicando pose fixa.")

#     def _load_robot_model(self):
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
#             if not success:
#                 rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#             rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo do robô do Parameter Server: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
#         damping_factor = 0.1 
#         self.kdl_solver_vel.setLambda(damping_factor)
        
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, 
#             self.kdl_solver_fk, 
#             self.kdl_solver_vel,
#             maxiter=200,   #maxiter=500
#             eps=1e-5   #eps=1e-3
#         )
        
#         self.q_init = kdl.JntArray(self.num_joints)

#     def _setup_ros_communication(self): 
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
        
#         # ATUALIZAÇÃO: Escuta a realidade do Gazebo a todo momento
#         rospy.Subscriber(self.JOINT_STATES_TOPIC, JointState, self.joint_states_callback, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)

#     def joint_states_callback(self, msg):
#         """Mantém a semente (q_init) perfeitamente alinhada com a realidade física."""
#         for i, joint_name in enumerate(self.JOINT_NAMES):
#             if joint_name in msg.name:
#                 idx = msg.name.index(joint_name)
#                 # Atualiza a semente em tempo real
#                 self.q_init[i] = msg.position[idx]
#         self.has_received_joints = True

#     def process_pose(self, pose_stamped_msg):
#         # Trava: Não tenta calcular se ainda não sabe onde o robô está
#         if not self.has_received_joints:
#             rospy.logwarn_throttle(2.0, "KDL: Aguardando semente física do Gazebo...")
#             return False

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        
#         q_out = kdl.JntArray(self.num_joints)
        
#         # A MÁGICA: O q_init aqui agora é a foto instantânea de onde o robô parou após o botão "Pick"!
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             self._publish_joint_command(q_out)
#             # A linha "self.q_init = q_out" foi removida daqui de propósito!
#             return True
#         else:
#             rospy.logwarn_throttle(1, "Alvo fora de alcance ou singularidade extrema. Código: %d", result)
#             return False

#     def pose_callback(self, data):
#         self.process_pose(data)
#         rospy.loginfo("Pose do Unity processada: %s", data.pose)

#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
        
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         point.time_from_start = rospy.Duration(0.01) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO GAZEBO ---")
        
#         pos = [0.3, 0.2, 0.5]
#         orient = [0.0, 1.57, 0.0] 
        
#         import tf.transformations
#         quat = tf.transformations.quaternion_from_euler(orient[0], orient[1], orient[2])
        
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         test_pose.orientation.x = quat[0]
#         test_pose.orientation.y = quat[1]
#         test_pose.orientation.z = quat[2]
#         test_pose.orientation.w = quat[3]
        
#         rate = rospy.Rate(1.0)
#         while not rospy.is_shutdown():
#             test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
#             solver.process_pose(test_stamped)
#             rate.sleep()
            
#     else:
#         rospy.spin()

# if __name__ == '__main__':
#     main()












# #!/usr/bin/env python3

# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState

# # --- VARIÁVEL GLOBAL PARA MODO DE TESTE ---
# # Altere para True para testar localmente no Gazebo sem o Unity.
# # Mantenha False para operar via Meta Quest / Unity.
# TEST_MODE = False
# # ----------------------------------------

# class KDLTeleopSolver:
#     """
#     Servidor de cinemática inversa usando KDL para controle do UR5.
#     Implementa Damped Least Squares (DLS) para estabilidade em singularidades.
#     """
    
#     # Parâmetros de Configuração
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
    
#     # Nomes das juntas do UR5 na ordem correta do Gazebo
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         """Inicializa o solucionador KDL e os componentes ROS."""
#         rospy.loginfo("Iniciando KDL Teleop Solver com amortecimento DLS...")
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
#         else:
#             rospy.loginfo("Modo ATIVO: Teste Manual Gazebo. Publicando pose fixa.")

#     def _load_robot_model(self):
#         """Carrega o modelo URDF e constrói a árvore KDL."""
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
#             if not success:
#                 rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#             rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo do robô do Parameter Server: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         """Inicializa os solucionadores de cinemática direta e inversa com DLS."""
#         # 1. Solucionador de Cinemática Direta (FK)
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
        
#         # 2. Solucionador de Velocidade Inversa Amortecido (WDLS)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
#         # Aplica o fator de amortecimento (Lambda). 
#         # Valor 0.1 é um excelente ponto de partida para o UR5.
#         damping_factor = 0.1 
#         self.kdl_solver_vel.setLambda(damping_factor)
        
#         # 3. Solucionador de Posição Inversa (Newton-Raphson)
#         # Usa o FK e o WDLS (amortecido) internamente para chegar na posição
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, 
#             self.kdl_solver_fk, 
#             self.kdl_solver_vel,
#             maxiter=500,
#             eps=1e-3
#         )
        
#         # Estado inicial das juntas (zeros)
#         self.q_init = kdl.JntArray(self.num_joints)


#     # def _setup_ros_communication(self):
#     #     """Configura publishers e subscribers ROS."""
#     #     self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
        
#     #     # Ouve o estado real do robô para alimentar a semente da IK
#     #     rospy.Subscriber('/joint_states', JointState, self.joint_states_callback, queue_size=1)
        
#     #     if not TEST_MODE:
#     #         self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)



#     # def joint_states_callback(self, msg):
#     #     """Atualiza a semente da IK com a posição real e atual das juntas do robô."""
#     #     # O Gazebo/UR5 publica juntas em ordem alfabética, o KDL precisa na ordem da cadeia cinemática.
#     #     for i, joint_name in enumerate(self.JOINT_NAMES):
#     #         if joint_name in msg.name:
#     #             idx = msg.name.index(joint_name)
#     #             # Atualiza a semente
#     #             self.q_init[i] = msg.position[idx]




#     def _setup_ros_communication(self): 
#         """Configura publishers e subscribers ROS."""
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)

#     def process_pose(self, pose_stamped_msg):
#         """
#         Função central de cálculo de IK. 
#         Converte a Pose cartesiana recebida em ângulos de junta.
#         """
#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        
#         q_out = kdl.JntArray(self.num_joints)
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0:  # IK Resolvido com Sucesso
#             self._publish_joint_command(q_out)
#             self.q_init = q_out  # Atualiza estado para o próximo cálculo ser mais rápido
#             return True
#         else:
#             # Com o DLS ativado, este erro só aparecerá se o alvo estiver MUITO fora da área de trabalho
#             rospy.logwarn_throttle(1, "Alvo fora de alcance ou singularidade extrema. Código: %d", result)
#             return False

#     def pose_callback(self, data):
#         """Disparado quando uma nova mensagem chega do Unity via ROS-TCP-Endpoint."""
#         self.process_pose(data)

#     def _publish_joint_command(self, joint_positions):
#         """Publica o comando de trajetória de juntas para o controlador do Gazebo."""
#         joint_positions_list = list(joint_positions)
        
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         # Define o tempo de alcance. 0.01s (100Hz) é ideal para teleoperação em tempo real
#         point.time_from_start = rospy.Duration(0.01) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)
#         rospy.loginfo_throttle(1, "Comando de juntas publicado: %s", joint_positions_list)

# def main():
#     """Função de inicialização do nó ROS."""
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO GAZEBO ---")
        
#         # Posição de teste (frente do robô)
#         pos = [0.3, 0.2, 0.5] # x, y, z em metros
#         orient = [0.0, 1.57, 0.0]  # Roll, Pitch, Yaw em radianos
        
#         import tf.transformations
#         quat = tf.transformations.quaternion_from_euler(orient[0], orient[1], orient[2])
        
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         test_pose.orientation.x = quat[0]
#         test_pose.orientation.y = quat[1]
#         test_pose.orientation.z = quat[2]
#         test_pose.orientation.w = quat[3]
        
#         rate = rospy.Rate(1.0) # 1 Hz para ver o robô se movendo devagar no teste
#         while not rospy.is_shutdown():
#             test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
#             solver.process_pose(test_stamped)
#             rate.sleep()
            
#     else:
#         # Modo de operação real (Aguardando Unity)
#         rospy.spin()

# if __name__ == '__main__':
#     main()
















# #!/usr/bin/env python3

# # Este nó ROS implementa um servidor de cinemática inversa usando a biblioteca KDL para controlar um braço robótico UR5.
# # Ele recebe uma pose alvo (posição + orientação) do Unity via ROS-TCP-Endpoint
# # O tool foi deslocado de 16cm para compensar a distância do tool0 até a ponta da garra do Robotiq 85, garantindo que a ponta dos dedos siga a pose desejada.

# import rospy
# import PyKDL as kdl
# import tf_conversions.posemath as pm
# from kdl_parser_py.urdf import treeFromUrdfModel
# from urdf_parser_py.urdf import URDF
# from geometry_msgs.msg import Pose, PoseStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState

# # --- VARIÁVEL GLOBAL PARA MODO DE TESTE ---
# # Altere para True para testar localmente no Gazebo sem o Unity.
# # Mantenha False para operar via Meta Quest / Unity.
# TEST_MODE = False
# # ----------------------------------------

# class KDLTeleopSolver:
#     """
#     Servidor de cinemática inversa usando KDL para controle do UR5.
#     Implementa Damped Least Squares (DLS) para estabilidade em singularidades.
#     """
    
#     # Parâmetros de Configuração
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     TCP_OFFSET_Z = 0.16  # Distância do tool0 até a ponta da garra (em metros) para o Robotiq 85 gripper
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/ur5/eff_joint_traj_controller/command'
    
#     # Nomes das juntas do UR5 na ordem correta do Gazebo
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#     ]

#     def __init__(self):
#         """Inicializa o solucionador KDL e os componentes ROS."""
#         rospy.loginfo("Iniciando KDL Teleop Solver com amortecimento DLS...")
        
#         self._load_robot_model()
#         self._initialize_kdl_solvers()
#         self._setup_ros_communication()
        
#         if not TEST_MODE:
#             rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
#         else:
#             rospy.loginfo("Modo ATIVO: Teste Manual Gazebo. Publicando pose fixa.")

#     def _load_robot_model(self):
#         """Carrega o modelo URDF e constrói a árvore KDL."""
#         try:
#             self.robot = URDF.from_parameter_server()
#             success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
#             if not success:
#                 rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
#                 raise Exception("Falha ao construir a árvore KDL.")
                
#             self.kdl_tree = kdl_tree_object
#             self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
#             self.num_joints = self.chain.getNrOfJoints()
#             rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
#         except Exception as e:
#             rospy.logerr("Erro ao carregar modelo do robô do Parameter Server: %s", str(e))
#             raise

#     def _initialize_kdl_solvers(self):
#         """Inicializa os solucionadores de cinemática direta e inversa com DLS."""
#         # 1. Solucionador de Cinemática Direta (FK)
#         self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
        
#         # 2. Solucionador de Velocidade Inversa Amortecido (WDLS)
#         self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
#         # Aplica o fator de amortecimento (Lambda). 
#         # Valor 0.1 é um excelente ponto de partida para o UR5.
#         damping_factor = 0.1 
#         self.kdl_solver_vel.setLambda(damping_factor)
        
#         # 3. Solucionador de Posição Inversa (Newton-Raphson)
#         # Usa o FK e o WDLS (amortecido) internamente para chegar na posição
#         self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
#             self.chain, 
#             self.kdl_solver_fk, 
#             self.kdl_solver_vel,
#             maxiter=500,
#             eps=1e-3
#         )
        
#         # Estado inicial das juntas (zeros)
#         self.q_init = kdl.JntArray(self.num_joints)

#     def _setup_ros_communication(self): 
#         """Configura publishers e subscribers ROS."""
#         self.command_pub = rospy.Publisher(self.COMMAND_TOPIC, JointTrajectory, queue_size=1)
        
#         if not TEST_MODE:
#             self.pose_sub = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)

#     def process_pose(self, pose_stamped_msg):
#         """
#         Função central de cálculo de IK. 
#         Converte a Pose cartesiana recebida em ângulos de junta.
#         """
#         # 1. Lê a pose que chegou do Unity (Representando a ponta dos dedos)
#         target_tcp_frame = pm.fromMsg(pose_stamped_msg.pose)
        
#         # --- INÍCIO DA COMPENSAÇÃO DE TCP ---
#         # Cria um frame recuando o valor do offset no próprio Eixo Z local da ferramenta
#         offset_frame = kdl.Frame(kdl.Rotation.RPY(0, 0, 0), kdl.Vector(0, 0, -self.TCP_OFFSET_Z))
        
#         # Multiplica o alvo pelo recuo. Isso gera a posição exata onde o tool0 deve ficar.
#         target_tool0_frame = target_tcp_frame * offset_frame
#         # --- FIM DA COMPENSAÇÃO ---
        
#         q_out = kdl.JntArray(self.num_joints)
        
#         # 2. Passa o target_tool0_frame (corrigido) para o Solver IK
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_tool0_frame, q_out)
        
#         if result >= 0:  # IK Resolvido com Sucesso
#             self._publish_joint_command(q_out)
#             self.q_init = q_out  # Atualiza estado para o próximo cálculo ser mais rápido
#             return True
#         else:
#             # Com o DLS ativado, este erro só aparecerá se o alvo estiver MUITO fora da área de trabalho
#             rospy.logwarn_throttle(1, "Alvo fora de alcance ou singularidade extrema. Código: %d", result)
#             return False

#     def pose_callback(self, data):
#         """Disparado quando uma nova mensagem chega do Unity via ROS-TCP-Endpoint."""
#         self.process_pose(data)

#     def _publish_joint_command(self, joint_positions):
#         """Publica o comando de trajetória de juntas para o controlador do Gazebo."""
#         joint_positions_list = list(joint_positions)
        
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
#         # Define o tempo de alcance. 0.01s (100Hz) é ideal para teleoperação em tempo real
#         point.time_from_start = rospy.Duration(0.01) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)
#         # Comentado para não poluir o terminal, ative se precisar debugar as juntas
#         # rospy.loginfo_throttle(1, "Comando de juntas publicado: %s", joint_positions_list)

# def main():
#     """Função de inicialização do nó ROS."""
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO GAZEBO ---")
        
#         # Posição de teste (frente do robô)
#         pos = [0.3, 0.2, 0.5] # x, y, z em metros
#         orient = [0.0, 1.57, 0.0]  # Roll, Pitch, Yaw em radianos
        
#         import tf.transformations
#         quat = tf.transformations.quaternion_from_euler(orient[0], orient[1], orient[2])
        
#         test_pose = Pose()
#         test_pose.position.x = pos[0]
#         test_pose.position.y = pos[1]
#         test_pose.position.z = pos[2]
        
#         test_pose.orientation.x = quat[0]
#         test_pose.orientation.y = quat[1]
#         test_pose.orientation.z = quat[2]
#         test_pose.orientation.w = quat[3]
        
#         rate = rospy.Rate(1.0) # 1 Hz para ver o robô se movendo devagar no teste
#         while not rospy.is_shutdown():
#             test_stamped = PoseStamped(header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), pose=test_pose)
#             solver.process_pose(test_stamped)
#             rate.sleep()
            
#     else:
#         # Modo de operação real (Aguardando Unity)
#         rospy.spin()

# if __name__ == '__main__':
#     main()







