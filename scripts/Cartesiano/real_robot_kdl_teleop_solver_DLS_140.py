#!/usr/bin/env python3
# Arquivo: real_robot_kdl_teleop_solver_DLS_140.py
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
TEST_MODE = True             # Ativa a rotina de teste manual
SEND_TO_ROBOT = False        # True: Envia comandos para o Robô Real | False: Apenas visualiza no terminal

# Escolha do modo de orientação do Alvo:
INPUT_AS_QUATERNION = True # True: Usa Quatérnio (x,y,z,w) | False: Usa Euler (r,p,y)

# Posição XYZ do Alvo (Comum para ambos os modos)
pos = [-0.42151787877082825, 0.031566303223371506, 0.3393116556107998]

# Opção A: Orientação em Euler (Usada se INPUT_AS_QUATERNION = False)
orient_euler = [3.14, 0.0, 0.0] # Roll, Pitch, Yaw (Garra apontada para baixo)

# Opção B: Orientação em Quatérnio (Usada se INPUT_AS_QUATERNION = True)
quat_input = [0.0, 0.003392765298485756, -0.9999942183494568, 0.0]  # Estrutura: [x, y, z, w]
# ==============================================================================

def normalizar_angulo(angulo):
    """
    Filtro Anti-Tornado: Força qualquer ângulo a ficar estritamente
    entre -π e +π (-180° a +180°), evitando giros mortais no robô real.
    """
    return math.atan2(math.sin(angulo), math.cos(angulo))

class KDLTeleopSolver:
    """
    Servidor de cinemática inversa usando KDL para controle do UR5 real.
    Implementa Damped Least Squares (DLS) e blindagem trigonométrica.
    """
    
    BASE_LINK = 'base_link'
    EE_LINK = 'tool0'
    POSE_TOPIC = 'unity/target_pose'
    COMMAND_TOPIC = '/scaled_pos_joint_traj_controller/command'
    JOINT_STATES_TOPIC = '/joint_states'
    
    JOINT_NAMES = [
        'shoulder_pan_joint', 
        'shoulder_lift_joint', 
        'elbow_joint',
        'wrist_1_joint', 
        'wrist_2_joint', 
        'wrist_3_joint'
    ]

    def __init__(self):
        rospy.loginfo("Iniciando KDL Teleop Solver com amortecimento DLS...")
        
        self.has_received_joints = False
        
        self._load_robot_model()
        self._initialize_kdl_solvers()
        self._setup_ros_communication()
        
        if not TEST_MODE:
            rospy.loginfo("Modo ATIVO: Recebendo poses do Unity no tópico '%s'", self.POSE_TOPIC)
        else:
            rospy.loginfo("Modo ATIVO: Teste Manual Ativo (SEND_TO_ROBOT = %s)", str(SEND_TO_ROBOT))

    def _load_robot_model(self):
        try:
            self.robot = URDF.from_parameter_server()
            success, kdl_tree_object = treeFromUrdfModel(self.robot)
            
            if not success:
                rospy.logerr("Falha Crítica: KDL não conseguiu construir a árvore a partir do URDF.")
                raise Exception("Falha ao construir a árvore KDL.")
                
            self.kdl_tree = kdl_tree_object
            self.chain = self.kdl_tree.getChain(self.BASE_LINK, self.EE_LINK)
            self.num_joints = self.chain.getNrOfJoints()
            rospy.loginfo("Modelo do robô carregado com sucesso. Juntas: %d", self.num_joints)
            
        except Exception as e:
            rospy.logerr("Erro ao carregar modelo do robô do Parameter Server: %s", str(e))
            raise

    def _initialize_kdl_solvers(self):
        self.kdl_solver_fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self.kdl_solver_vel = kdl.ChainIkSolverVel_wdls(self.chain)
        
        damping_factor = 1   # Fator de amortecimento para o DLS
        self.kdl_solver_vel.setLambda(damping_factor)
        
        self.kdl_solver_pos = kdl.ChainIkSolverPos_NR(
            self.chain, 
            self.kdl_solver_fk, 
            self.kdl_solver_vel,
            maxiter=50,   
            eps=1e-3   
        )
        
        self.q_init = kdl.JntArray(self.num_joints)
        
        # --- NOVAS VARIÁVEIS DE MEMÓRIA MUSCULAR ---
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
            rospy.logwarn_throttle(2.0, "KDL: Aguardando semente física do hardware real...")
            return False

        target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        q_out = kdl.JntArray(self.num_joints)
        
        # --- A MÁGICA DA MEMÓRIA ESTÁ AQUI ---
        # Usa o robô real no 1º milissegundo. Depois, usa a matemática anterior para evitar o FLIP.
        seed = self.q_init if self.is_first_ik else self.last_q_out
        
        result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
        if result >= 0: 
            self.last_q_out = q_out      # Salva a memória para o próximo ciclo
            self.is_first_ik = False     # Desliga a semente ruidosa do hardware

            # --- BLINDAGEM MATEMÁTICA ---
            juntas_normalizadas = [normalizar_angulo(q_out[i]) for i in range(self.num_joints)]
            juntas_graus = [math.degrees(j) for j in juntas_normalizadas]
            
            print("\n" + "="*60)
            print("[KDL SOLVER] Solução de Cinemática Inversa blindada e congelada!")
            print("="*60)
            for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
                print(f" -> {nome}: {angulo:.2f}°")
            print("="*60)

            if SEND_TO_ROBOT:
                self._publish_joint_command(juntas_normalizadas)
                print("[INFO] Comando enviado com segurança ao Robô Real.")
            else:
                print("[INFO] Modo Visualização Ativo: Comando NÃO enviado ao Robô Real.")
                
            return True
        else:
            rospy.logwarn_throttle(1, "Alvo fora de alcance ou singularidade extrema. Código: %d", result)
            return False

    def pose_callback(self, data):
        self.process_pose(data)

    def autonomous_pose_callback(self, pose_stamped_msg):
        if not self.has_received_joints:
            return

        target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
        q_out = kdl.JntArray(self.num_joints)
        
        # Aplica a mesma memória muscular aos menus autônomos para consistência geométrica
        seed = self.q_init if self.is_first_ik else self.last_q_out
        result = self.kdl_solver_pos.CartToJnt(seed, target_pose_kdl, q_out)
        
        if result >= 0: 
            self.last_q_out = q_out
            self.is_first_ik = False

            # --- BLINDAGEM MATEMÁTICA PARA OS MENUS AUTÔNOMOS ---
            juntas_normalizadas = [normalizar_angulo(q_out[i]) for i in range(self.num_joints)]
            
            js_msg = JointState()
            js_msg.name = self.JOINT_NAMES + ['finger_joint']
            pos_list = juntas_normalizadas
            pos_list.append(0.0) # Garante o tamanho do array com um valor neutro para a garra
            js_msg.position = pos_list
            self.quintic_pub.publish(js_msg)

    def _publish_joint_command(self, joint_positions):
        joint_positions_list = list(joint_positions)
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.JOINT_NAMES
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions_list
        
        # --- FREIO DE SOFTWARE PARA TELEOPERAÇÃO SEGURA --- (o original era 0.04)
        point.time_from_start = rospy.Duration(1.0) 
        
        traj_msg.points.append(point)
        self.command_pub.publish(traj_msg)

def main():
    rospy.init_node('kdl_teleop_solver_node', anonymous=True)
    solver = KDLTeleopSolver()
    
    if TEST_MODE:
        rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO ROBÔ REAL ---")
        
        test_pose = Pose()
        test_pose.position.x = pos[0]
        test_pose.position.y = pos[1]
        test_pose.position.z = pos[2]
        
        if INPUT_AS_QUATERNION:
            rospy.loginfo("[TEST_MODE] Utilizando orientação via QUATÉRNIO direto.")
            test_pose.orientation.x = quat_input[0]
            test_pose.orientation.y = quat_input[1]
            test_pose.orientation.z = quat_input[2]
            test_pose.orientation.w = quat_input[3]
        else:
            rospy.loginfo("[TEST_MODE] Utilizando orientação via ÂNGULOS DE EULER.")
            import tf.transformations
            quat = tf.transformations.quaternion_from_euler(orient_euler[0], orient_euler[1], orient_euler[2])
            test_pose.orientation.x = quat[0]
            test_pose.orientation.y = quat[1]
            test_pose.orientation.z = quat[2]
            test_pose.orientation.w = quat[3]
        
        def timer_callback(event):
            test_stamped = PoseStamped(
                header=Header(frame_id=solver.BASE_LINK, stamp=rospy.Time.now()), 
                pose=test_pose
            )
            solver.process_pose(test_stamped)

        rospy.Timer(rospy.Duration(1.0), timer_callback)
        rospy.spin()
            
    else:
        rospy.spin()

if __name__ == '__main__':
    main()












# #!/usr/bin/env python3
# # Arquivo: real_robot_kdl_teleop_solver_DLS_140.py
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
# SEND_TO_ROBOT = True        # True: Envia comandos para o Robô Real | False: Apenas visualiza no terminal

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
#     entre -π e +π (-180° a +180°), evitando giros mortais no robô real.
#     """
#     return math.atan2(math.sin(angulo), math.cos(angulo))

# class KDLTeleopSolver:
#     """
#     Servidor de cinemática inversa usando KDL para controle do UR5 real.
#     Implementa Damped Least Squares (DLS) e blindagem trigonométrica.
#     """
    
#     BASE_LINK = 'base_link'
#     EE_LINK = 'tool0'
#     POSE_TOPIC = 'unity/target_pose'
#     COMMAND_TOPIC = '/scaled_pos_joint_traj_controller/command'
#     JOINT_STATES_TOPIC = '/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 
#         'shoulder_lift_joint', 
#         'elbow_joint',
#         'wrist_1_joint', 
#         'wrist_2_joint', 
#         'wrist_3_joint'
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
#             rospy.loginfo("Modo ATIVO: Teste Manual Ativo (SEND_TO_ROBOT = %s)", str(SEND_TO_ROBOT))

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
#             rospy.logwarn_throttle(2.0, "KDL: Aguardando semente física do hardware real...")
#             return False

#         target_pose_kdl = pm.fromMsg(pose_stamped_msg.pose)
#         q_out = kdl.JntArray(self.num_joints)
        
#         result = self.kdl_solver_pos.CartToJnt(self.q_init, target_pose_kdl, q_out)
        
#         if result >= 0: 
#             # --- BLINDAGEM MATEMÁTICA ---
#             juntas_normalizadas = [normalizar_angulo(q_out[i]) for i in range(self.num_joints)]
#             juntas_graus = [math.degrees(j) for j in juntas_normalizadas]
            
#             print("\n" + "="*60)
#             print("[KDL SOLVER] Solução de Cinemática Inversa blindada!")
#             print("="*60)
#             for nome, angulo in zip(self.JOINT_NAMES, juntas_graus):
#                 print(f" -> {nome}: {angulo:.2f}°")
#             print("="*60)

#             if SEND_TO_ROBOT:
#                 self._publish_joint_command(juntas_normalizadas)
#                 print("[INFO] Comando enviado com segurança ao Robô Real.")
#             else:
#                 print("[INFO] Modo Visualização Ativo: Comando NÃO enviado ao Robô Real.")
                
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
#             # --- BLINDAGEM MATEMÁTICA PARA OS MENUS AUTÔNOMOS ---
#             juntas_normalizadas = [normalizar_angulo(q_out[i]) for i in range(self.num_joints)]
            
#             js_msg = JointState()
#             js_msg.name = self.JOINT_NAMES + ['finger_joint']
#             pos_list = juntas_normalizadas
#             pos_list.append(0.0) # Garante o tamanho do array com um valor neutro para a garra
#             js_msg.position = pos_list
#             self.quintic_pub.publish(js_msg)

#     def _publish_joint_command(self, joint_positions):
#         joint_positions_list = list(joint_positions)
#         traj_msg = JointTrajectory()
#         traj_msg.joint_names = self.JOINT_NAMES
        
#         point = JointTrajectoryPoint()
#         point.positions = joint_positions_list
        
#         # --- FREIO DE SOFTWARE PARA TELEOPERAÇÃO SEGURA --- o original era 0.04 e foi modificado para 1.0 por segurança no robô real
#         point.time_from_start = rospy.Duration(1.0) 
        
#         traj_msg.points.append(point)
#         self.command_pub.publish(traj_msg)

# def main():
#     rospy.init_node('kdl_teleop_solver_node', anonymous=True)
#     solver = KDLTeleopSolver()
    
#     if TEST_MODE:
#         rospy.loginfo("--- INICIANDO ROTINA DE TESTE NO ROBÔ REAL ---")
        
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
#     COMMAND_TOPIC = '/scaled_pos_joint_traj_controller/command'
#     JOINT_STATES_TOPIC = '/joint_states'
    
#     JOINT_NAMES = [
#         'shoulder_pan_joint', 
#         'shoulder_lift_joint', 
#         'elbow_joint',
#         'wrist_1_joint', 
#         'wrist_2_joint', 
#         'wrist_3_joint'
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







