#!/usr/bin/env python3
# gazebo_test_gripper.py
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def send_gazebo_gripper(pub, position_rad):
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = ["finger_joint"]
    
    point = JointTrajectoryPoint()
    point.positions = [position_rad]
    point.time_from_start = rospy.Duration(0.5) # Transição em meio segundo
    
    msg.points.append(point)
    pub.publish(msg)

rospy.init_node('gazebo_test_gripper_node')
# Tópico da garra simulada no Gazebo
pub = rospy.Publisher('/ur5/gripper_controller/command', JointTrajectory, queue_size=1)

rospy.sleep(1.0)  

rospy.loginfo("1. Abrindo a garra completamente no Gazebo (0.0 rad)...")
send_gazebo_gripper(pub, 0.0)
rospy.sleep(3.0)

rospy.loginfo("2. Fechando a garra parcialmente (0.4 rad, equivalente ao byte 128)...")
send_gazebo_gripper(pub, 0.4)
rospy.sleep(3.0)

rospy.loginfo("3. Abrindo novamente (0.0 rad)...")
send_gazebo_gripper(pub, 0.0)
rospy.sleep(2.0)

rospy.loginfo("Teste da garra no simulador concluído!")