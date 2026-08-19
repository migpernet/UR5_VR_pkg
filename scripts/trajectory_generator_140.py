#!/usr/bin/env python3

# trajectory_generator.py
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class QuinticPolynomial:
    """
    Classe para calcular os coeficientes e a posição ao longo do tempo (t)
    para uma trajetória suave (zero velocidade e aceleração nas bordas).
    """

    def __init__(self, theta_init, theta_final, T):
        """
        Calcula os coeficientes a0 a a5.
        :param theta_init: Posição inicial (radianos).
        :param theta_final: Posição final (radianos).
        :param T: Tempo total da trajetória (segundos).
        """
        self.theta_init = theta_init
        self.theta_final = theta_final
        self.T = T
        
        delta_theta = theta_final - theta_init
        
        # Coeficientes
        self.a0 = theta_init
        self.a1 = 0.0
        self.a2 = 0.0
        
        T3 = T**3
        T4 = T**4
        T5 = T**5
        
        # Coeficientes do polinômio quíntuplo (posição)
        self.a3 = 10.0 * delta_theta / T3
        self.a4 = -15.0 * delta_theta / T4
        self.a5 = 6.0 * delta_theta / T5

    def get_position(self, t):
        """
        Retorna a posição da junta no tempo t.
        """
        if t < 0:
            return self.theta_init
        if t >= self.T:
            return self.theta_final
            
        t2 = t*t
        t3 = t*t2
        t4 = t*t3
        t5 = t*t4
        
        # theta(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        return self.a0 + self.a1*t + self.a2*t2 + self.a3*t3 + self.a4*t4 + self.a5*t5

def generate_smooth_trajectory(initial_pose, final_pose, T, frequency_hz):
    """
    Gera uma JointTrajectoryMsg completa a partir das poses inicial e final.

    :param initial_pose: Lista de posições iniciais de todas as 7 juntas (radianos).
    :param final_pose: Lista de posições finais (radianos).
    :param T: Tempo total de movimento (segundos).
    :param frequency_hz: Frequência de publicação (Hz).
    :return: JointTrajectory preenchida (sem nomes de juntas) ou None.
    """
    num_juntas = len(initial_pose)
    num_pontos = int(T * frequency_hz) + 1 # Garante pelo menos o ponto final
    time_step = T / (num_pontos - 1.0)
    
    # 1. Cria os planejadores (polinômios) para cada junta
    planners = []
    for init, final in zip(initial_pose, final_pose):
        planners.append(QuinticPolynomial(init, final, T))
        
    trajectory = JointTrajectory()
    
    # 2. Gera os pontos de trajetória
    for i in range(num_pontos):
        t = i * time_step
        point = JointTrajectoryPoint()
        point.positions = [p.get_position(t) for p in planners]
        point.time_from_start = rospy.Duration(t)
        
        trajectory.points.append(point)
        
    return trajectory