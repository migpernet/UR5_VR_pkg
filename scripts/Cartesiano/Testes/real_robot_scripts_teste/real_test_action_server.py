#!/usr/bin/env python3
# real_test_action_server.py
# giro maior de 45 graus no pulso 3, com retorno exato para a posição inicial
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
]

current_joints = []

def js_cb(msg):
    global current_joints
    if all(name in msg.name for name in JOINT_NAMES):
        current_joints = [msg.position[msg.name.index(n)] for n in JOINT_NAMES]

rospy.init_node('test_action_server_node')
rospy.Subscriber('/joint_states', JointState, js_cb)

rospy.loginfo("Aguardando /joint_states...")
while not rospy.is_shutdown() and len(current_joints) == 0:
    rospy.sleep(0.1)

# "Fotografa" a pose exata em que o robô está parado para garantir a volta perfeita
initial_joints = list(current_joints)

action_topic = '/scaled_pos_joint_traj_controller/follow_joint_trajectory'
client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)

rospy.loginfo(f"Conectando ao Action Server: {action_topic} ...")
if not client.wait_for_server(timeout=rospy.Duration(5.0)):
    rospy.logerr("Falha: Action Server não encontrado!")
    exit(1)

rospy.loginfo("Action Server CONECTADO! Enviando trajetória de vai-e-volta...")

traj = JointTrajectory()
traj.header.stamp = rospy.Time.now()
traj.joint_names = JOINT_NAMES

# Ponto 1: Posição atual (Início)
p1 = JointTrajectoryPoint(positions=initial_joints, time_from_start=rospy.Duration(0.5))

# Ponto 2: Desloca o pulso 3 em +0.8 rad (~45 graus) no tempo de 2.5 segundos
target_joints = list(initial_joints)
target_joints[5] += 0.8
p2 = JointTrajectoryPoint(positions=target_joints, time_from_start=rospy.Duration(2.5))

# Ponto 3: Congela na posição deslocada por 1 segundo (feedback visual)
p3 = JointTrajectoryPoint(positions=target_joints, time_from_start=rospy.Duration(3.5))

# Ponto 4: Retorna de forma precisa para a posição INICIAL EXATA
p4 = JointTrajectoryPoint(positions=initial_joints, time_from_start=rospy.Duration(6.0))

traj.points = [p1, p2, p3, p4]

goal = FollowJointTrajectoryGoal(trajectory=traj)
client.send_goal(goal)
client.wait_for_result()

rospy.loginfo(f"Resultado do Action Server: {client.get_state()} (3 = SUCESSO)")
rospy.loginfo("Movimento concluído! O robô retornou exatamente para a posição inicial.")







# #!/usr/bin/env python3
# # real_test_action_server.py
# import rospy
# import actionlib
# from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from sensor_msgs.msg import JointState

# JOINT_NAMES = [
#     "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
#     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
# ]

# current_joints = []

# def js_cb(msg):
#     global current_joints
#     if all(name in msg.name for name in JOINT_NAMES):
#         current_joints = [msg.position[msg.name.index(n)] for n in JOINT_NAMES]

# rospy.init_node('test_action_server_node')
# rospy.Subscriber('/joint_states', JointState, js_cb)

# rospy.loginfo("Aguardando /joint_states...")
# while not rospy.is_shutdown() and len(current_joints) == 0:
#     rospy.sleep(0.1)

# action_topic = '/scaled_pos_joint_traj_controller/follow_joint_trajectory'
# client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)

# rospy.loginfo(f"Conectando ao Action Server: {action_topic} ...")
# if not client.wait_for_server(timeout=rospy.Duration(5.0)):
#     rospy.logerr("Falha: Action Server não encontrado!")
#     exit(1)

# rospy.loginfo("Action Server CONECTADO! Enviando micro-trajetória de teste...")

# traj = JointTrajectory()
# traj.header.stamp = rospy.Time.now()
# traj.joint_names = JOINT_NAMES

# # Ponto 1: Posição atual
# p1 = JointTrajectoryPoint(positions=list(current_joints), time_from_start=rospy.Duration(0.5))

# # Ponto 2: Desloca o pulso 3 em +0.08 rad (~4.5 graus) com retorno suave
# target_joints = list(current_joints)
# target_joints[5] += 0.08
# p2 = JointTrajectoryPoint(positions=target_joints, time_from_start=rospy.Duration(3.0))

# traj.points = [p1, p2]

# goal = FollowJointTrajectoryGoal(trajectory=traj)
# client.send_goal(goal)
# client.wait_for_result()
# rospy.loginfo(f"Resultado do Action Server: {client.get_state()} (3 = SUCESSO)")







