#!/usr/bin/env python3
# real_test_robotiq_gripper.py
import rospy
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output

def send_gripper(pub, position, speed=150, force=150):
    cmd = Robotiq2FGripper_robot_output()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rATR = 0
    cmd.rPR = int(max(0, min(255, position)))
    cmd.rSP = speed
    cmd.rFR = force
    pub.publish(cmd)

rospy.init_node('test_robotiq_gripper_node')
pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=1)

rospy.sleep(1.0)  # Aguarda estabelecimento da conexão com o driver da garra

rospy.loginfo("1. Abrindo a garra completamente (rPR = 0)...")
send_gripper(pub, 0)
rospy.sleep(3.0)

rospy.loginfo("2. Fechando a garra parcialmente (rPR = 128)...")
send_gripper(pub, 128)
rospy.sleep(3.0)

rospy.loginfo("3. Abrindo novamente (rPR = 0)...")
send_gripper(pub, 0)
rospy.sleep(2.0)

rospy.loginfo("Teste da garra concluído!")