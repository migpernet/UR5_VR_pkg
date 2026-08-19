#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64

def callback(data):
    # Apenas retransmite o timestamp que recebeu do Unity
    # data.data contém o tempo que o Unity enviou
    pub.publish(data)

rospy.init_node('latency_echo_node')
# Tópico de retorno (Echo)
pub = rospy.Publisher('/latency_echo', Float64, queue_size=10)
# Tópico de entrada (Ping)
rospy.Subscriber('/latency_ping', Float64, callback)

rospy.loginfo("Nó de teste de latência iniciado. Aguardando pings do Unity...")
rospy.spin()