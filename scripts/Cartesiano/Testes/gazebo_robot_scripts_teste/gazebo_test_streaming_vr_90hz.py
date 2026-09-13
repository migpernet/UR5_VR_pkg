#!/usr/bin/env python3
# gazebo_test_streaming_vr_90hz.py
import rospy
import math
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

rospy.init_node('gazebo_test_streaming_vr_node')
rospy.Subscriber('/ur5/joint_states', JointState, js_cb)

rospy.loginfo("Aguardando leitura de /ur5/joint_states...")
while not rospy.is_shutdown() and len(current_joints) == 0:
    rospy.sleep(0.1)

# Tópico do Publisher de streaming direto no Gazebo
pub_topic = '/ur5/eff_joint_traj_controller/command'
pub = rospy.Publisher(pub_topic, JointTrajectory, queue_size=1)

rate = rospy.Rate(90)  # Simula a taxa do headset VR (90 Hz)
initial_joints = list(current_joints)
rospy.loginfo("Iniciando streaming suave senoidal a 90 Hz no Gazebo (Ctrl+C para parar)...")

start_time = rospy.Time.now()

while not rospy.is_shutdown():
    elapsed = (rospy.Time.now() - start_time).to_sec()
    
    # Onda senoidal de baixa amplitude: oscila +/- 2 graus (0.035 rad) em torno da pose atual
    offset = 0.035 * math.sin(2.0 * math.pi * 0.5 * elapsed)
    
    cmd_joints = list(initial_joints)
    cmd_joints[5] += offset  # Aplica no wrist_3_joint
    
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = JOINT_NAMES
    
    point = JointTrajectoryPoint()
    point.positions = cmd_joints
    point.velocities = [0.0] * 6
    point.time_from_start = rospy.Duration(0.04) 
    
    msg.points.append(point)
    pub.publish(msg)
    rate.sleep()