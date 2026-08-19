#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2

class ColorFixer:
    def __init__(self):
        # Escuta a nuvem original com cores trocadas
        self.sub = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.callback, queue_size=1)
        # Publica a nuvem corrigida
        self.pub = rospy.Publisher('/camera/depth/color/points_fixed', PointCloud2, queue_size=1)
        rospy.loginfo("Nó Corretor de Cores Iniciado! Publicando em /camera/depth/color/points_fixed")

    def callback(self, msg):
        if not msg.fields:
            return

        # Encontra onde a cor (RGB) está escondida dentro dos dados do ponto
        rgb_offset = None
        for f in msg.fields:
            if f.name == 'rgb' or f.name == 'rgba':
                rgb_offset = f.offset
                break

        if rgb_offset is None:
            self.pub.publish(msg)
            return

        # Converte os dados brutos em uma matriz NumPy para processamento ultrarrápido
        buf = np.frombuffer(msg.data, dtype=np.uint8).copy()
        buf = buf.reshape(-1, msg.point_step)

        # A Mágica: Troca o byte do Azul (offset 0) com o do Vermelho (offset 2)
        temp = buf[:, rgb_offset].copy()
        buf[:, rgb_offset] = buf[:, rgb_offset + 2]
        buf[:, rgb_offset + 2] = temp

        # Empacota de volta e publica
        msg.data = buf.tobytes()
        self.pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('pointcloud_color_fixer')
    ColorFixer()
    rospy.spin()
