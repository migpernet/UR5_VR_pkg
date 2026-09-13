#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
UNIFIED TRAC-IK TELEOP SOLVER (HARDWARE REAL)
===============================================================================

Arquitetura (Robô Físico UR5):

    Unity VR
       |
       |  /unity/target_pose
       v
+-----------------------------+
|       TRAC-IK Solver        |
|                             |
|  - Distance mode            |
|  - Seed físico / temporal   |
|  - Injeção URDF dinâmica    |
|  - Anti-unwinding           |
|  - Prevenção Protective Stop|
+-----------------------------+
       |
       | /ur5/planned_trajectory
       v
+-----------------------------+
|   Robot Commander (Híbrido) |
+-----------------------------+
       |
       v
 UR5 Real (Action Server/Pub)

===============================================================================
"""

import rospy
import math
import time
import threading
import xml.etree.ElementTree as ET

from trac_ik_python.trac_ik import IK

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


# =============================================================================
# CONFIGURAÇÃO PRINCIPAL
# =============================================================================

TEST_MODE = False
SEND_TO_HARDWARE = True  # Modificado para clareza no robô real

INPUT_AS_QUATERNION = True

BASE_LINK = "base_link"
EE_LINK = "tool0"

POSE_TOPIC = "unity/target_pose"
AUTONOMOUS_POSE_TOPIC = "unity/target_pose_autonomous"

# Tópico que alimenta o seu nó RobotCommanderNode (Híbrido)
COMMAND_TOPIC = "/ur5/planned_trajectory"

# [ROBÔ REAL]: O driver da UR publica na raiz
JOINT_STATES_TOPIC = "/joint_states"

TARGET_JOINTS_TOPIC = "/unity/target_joints"


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]


# =============================================================================
# TRAC-IK
# =============================================================================

TRACIK_TIMEOUT = 0.005
TRACIK_EPSILON = 1e-4
TRACIK_SOLVE_TYPE = "Distance"


# =============================================================================
# SEGURANÇA / CONTROLE
# =============================================================================

# Threshold de detecção de atuação do menu (11.5 graus)
MENU_DRIFT_THRESHOLD = 0.20  

# Velocidade máxima para cálculo do amortecedor dinâmico
MAX_RAD_PER_SEC = 2.0

# [ROBÔ REAL]: Piso de 200ms vital para evitar Protective Stops
MIN_COMMAND_DURATION = 0.20

DEFAULT_MIN = -2.0 * math.pi
DEFAULT_MAX = +2.0 * math.pi


# =============================================================================
# LIMITES ARTICULARES (GESSO VIRTUAL)
# =============================================================================

JOINT_MIN = [
    DEFAULT_MIN,   # shoulder_pan
    -2.3,          # shoulder_lift (Elbow Up Seguro)
    -math.pi,      # elbow
    DEFAULT_MIN,   # wrist_1
    0.1,           # wrist_2 (Anti-Singularidade do Punho)
    DEFAULT_MIN    # wrist_3
]

JOINT_MAX = [
    DEFAULT_MAX,   # shoulder_pan
    0.0,           # shoulder_lift
    -0.1,          # elbow
    DEFAULT_MAX,   # wrist_1
    3.0,           # wrist_2
    DEFAULT_MAX    # wrist_3
]


# =============================================================================
# AUXILIARES MATEMÁTICOS
# =============================================================================

def is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def all_finite(values):
    return all(is_finite(v) for v in values)


def circular_difference(target, current):
    return math.atan2(
        math.sin(target - current),
        math.cos(target - current)
    )


def resolver_salto_angular(angulo_alvo, angulo_atual):
    delta = circular_difference(
        angulo_alvo,
        angulo_atual
    )
    resultado = angulo_atual + delta

    while resultado > 2.0 * math.pi:
        resultado -= 2.0 * math.pi

    while resultado < -2.0 * math.pi:
        resultado += 2.0 * math.pi

    return resultado


def distancia_articular(q_a, q_b):
    if len(q_a) != len(q_b):
        return float("inf")

    deltas = [
        abs(circular_difference(a, b))
        for a, b in zip(q_a, q_b)
    ]

    return max(deltas)


# =============================================================================
# NÓ PRINCIPAL
# =============================================================================

class UnifiedTRACIKTeleopSolver(object):

    def __init__(self):
        rospy.loginfo("================================================")
        rospy.loginfo(" UNIFIED TRAC-IK TELEOP SOLVER (HARDWARE REAL)")
        rospy.loginfo("================================================")

        self.has_received_joints = False
        self.q_init = [0.0] * 6
        self.last_q_out = [0.0] * 6
        self.is_first_ik = True
        self.ik_lock = threading.Lock()

        self.ik_count = 0
        self.ik_success = 0
        self.ik_failure = 0
        self.last_ik_time = 0.0

        self._initialize_trac_ik()
        self._setup_ros()

        rospy.loginfo("TRAC-IK solver inicializado com sucesso para o Robô Real.")

    def _initialize_trac_ik(self):
        rospy.loginfo("Inicializando TRAC-IK...")

        try:
            urdf_string = rospy.get_param("/robot_description")

            if not urdf_string:
                raise RuntimeError("Parâmetro /robot_description está vazio.")

            modified_urdf = self._inject_joint_limits(urdf_string)

            self.ik_solver = IK(
                BASE_LINK,
                EE_LINK,
                urdf_string=modified_urdf,
                timeout=TRACIK_TIMEOUT,
                epsilon=TRACIK_EPSILON,
                solve_type=TRACIK_SOLVE_TYPE
            )

            rospy.loginfo("TRAC-IK encontrou %d joints.", self.ik_solver.number_of_joints)
            rospy.loginfo("TRAC-IK joint names: %s", str(self.ik_solver.joint_names))

            if self.ik_solver.number_of_joints != len(JOINT_NAMES):
                raise RuntimeError(
                    "Número de joints incompatível: TRAC-IK=%d, esperado=%d"
                    % (self.ik_solver.number_of_joints, len(JOINT_NAMES))
                )

            if list(self.ik_solver.joint_names) != JOINT_NAMES:
                raise RuntimeError(
                    "ORDEM DOS JOINTS INCOMPATÍVEL!\nEsperado: %s\nTRAC-IK:  %s"
                    % (JOINT_NAMES, self.ik_solver.joint_names)
                )

            self.ik_solver.set_joint_limits(JOINT_MIN, JOINT_MAX)
            lower, upper = self.ik_solver.get_joint_limits()

            for i in range(len(JOINT_NAMES)):
                if abs(lower[i] - JOINT_MIN[i]) > 1e-6:
                    raise RuntimeError("Lower limit incorreto no joint %s." % JOINT_NAMES[i])
                if abs(upper[i] - JOINT_MAX[i]) > 1e-6:
                    raise RuntimeError("Upper limit incorreto no joint %s." % JOINT_NAMES[i])

            rospy.loginfo(
                "TRAC-IK configurado: timeout=%.4f s | epsilon=%.1e | solve_type=%s",
                TRACIK_TIMEOUT, TRACIK_EPSILON, TRACIK_SOLVE_TYPE
            )

        except Exception as exc:
            rospy.logerr("Falha crítica ao inicializar TRAC-IK: %s", str(exc))
            raise

    def _inject_joint_limits(self, urdf_string):
        try:
            root = ET.fromstring(urdf_string)
        except Exception as exc:
            raise RuntimeError("Não foi possível interpretar /robot_description: %s" % str(exc))

        joint_map = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            if name in JOINT_NAMES:
                joint_map[name] = joint

        missing = [name for name in JOINT_NAMES if name not in joint_map]
        if missing:
            raise RuntimeError("Joints ausentes no URDF: %s" % str(missing))

        for i, joint_name in enumerate(JOINT_NAMES):
            joint = joint_map[joint_name]
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")

            limit.set("lower", str(JOINT_MIN[i]))
            limit.set("upper", str(JOINT_MAX[i]))

        modified_urdf = ET.tostring(root, encoding="unicode")
        rospy.loginfo("Limites virtuais TRAC-IK injetados no URDF.")
        return modified_urdf

    def _setup_ros(self):
        self.command_pub = rospy.Publisher(COMMAND_TOPIC, JointTrajectory, queue_size=1)
        self.target_joints_pub = rospy.Publisher(TARGET_JOINTS_TOPIC, JointState, queue_size=1)

        rospy.Subscriber(JOINT_STATES_TOPIC, JointState, self._joint_state_callback, queue_size=1)
        rospy.Subscriber(POSE_TOPIC, PoseStamped, self._pose_callback, queue_size=1)
        rospy.Subscriber(AUTONOMOUS_POSE_TOPIC, PoseStamped, self._autonomous_pose_callback, queue_size=1)

        rospy.loginfo("Subscriber: %s", JOINT_STATES_TOPIC)
        rospy.loginfo("Subscriber: %s", POSE_TOPIC)
        rospy.loginfo("Subscriber: %s", AUTONOMOUS_POSE_TOPIC)
        rospy.loginfo("Publisher: %s", COMMAND_TOPIC)

    def _joint_state_callback(self, msg):
        q_new = [None] * len(JOINT_NAMES)

        for expected_index, expected_name in enumerate(JOINT_NAMES):
            found = False
            for i, name in enumerate(msg.name):
                if name == expected_name:
                    if i < len(msg.position):
                        q_new[expected_index] = float(msg.position[i])
                        found = True
                    break
            if not found:
                break

        if any(q is None for q in q_new):
            return

        if not all_finite(q_new):
            rospy.logwarn_throttle(2.0, "JointState contém valor NaN/Inf.")
            return

        self.q_init = q_new
        self.has_received_joints = True

    def _extract_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]

        if not all_finite(values):
            rospy.logwarn("Pose descartada: contém NaN/Inf.")
            return None

        q_norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)

        if not math.isfinite(q_norm):
            rospy.logwarn("Pose descartada: norma do quaternion inválida.")
            return None

        if q_norm < 1e-9:
            rospy.logwarn("Pose descartada: quaternion praticamente nulo.")
            return None

        qx = q.x / q_norm
        qy = q.y / q_norm
        qz = q.z / q_norm
        qw = q.w / q_norm

        return (float(p.x), float(p.y), float(p.z), float(qx), float(qy), float(qz), float(qw))

    def _check_external_drift(self):
        if self.is_first_ik:
            return

        drift = distancia_articular(self.q_init, self.last_q_out)
        if drift > MENU_DRIFT_THRESHOLD:
            rospy.logwarn(
                "DRIFT EXTERNO detectado: %.3f rad (threshold %.3f). Resetando seed para posição física.",
                drift, MENU_DRIFT_THRESHOLD
            )
            self.is_first_ik = True

    def _validate_solution(self, solution, seed):
        if solution is None:
            return False, "TRAC-IK retornou None"

        if len(solution) != len(JOINT_NAMES):
            return False, "número de joints incorreto"

        if not all_finite(solution):
            return False, "solução contém NaN/Inf"

        for i, q in enumerate(solution):
            if q < JOINT_MIN[i] - 1e-6:
                return False, "joint %s abaixo do limite: %.6f < %.6f" % (JOINT_NAMES[i], q, JOINT_MIN[i])
            if q > JOINT_MAX[i] + 1e-6:
                return False, "joint %s acima do limite: %.6f > %.6f" % (JOINT_NAMES[i], q, JOINT_MAX[i])

        for i in range(len(solution)):
            delta = abs(circular_difference(solution[i], seed[i]))
            if delta > math.pi + 1e-6:
                return False, "salto angular suspeito no joint %s: %.3f rad" % (JOINT_NAMES[i], delta)

        return True, "OK"

    def _solve_ik(self, pose, seed):
        px, py, pz, qx, qy, qz, qw = pose
        start = time.monotonic()

        try:
            with self.ik_lock:
                solution = self.ik_solver.get_ik(list(seed), px, py, pz, qx, qy, qz, qw)
        except Exception as exc:
            elapsed = time.monotonic() - start
            self.last_ik_time = elapsed
            self.ik_failure += 1
            rospy.logerr("Exceção no TRAC-IK após %.3f ms: %s", elapsed * 1000.0, str(exc))
            return None

        elapsed = time.monotonic() - start
        self.last_ik_time = elapsed
        self.ik_count += 1

        if solution is None:
            self.ik_failure += 1
            rospy.logwarn_throttle(1.0, "TRAC-IK não encontrou solução. tempo=%.3f ms", elapsed * 1000.0)
            return None

        solution = [float(q) for q in solution]
        corrected_solution = []

        for i in range(len(solution)):
            q_corrected = resolver_salto_angular(solution[i], seed[i])
            corrected_solution.append(q_corrected)

        solution = corrected_solution
        valid, reason = self._validate_solution(solution, seed)

        if not valid:
            self.ik_failure += 1
            rospy.logwarn("Solução TRAC-IK rejeitada: %s", reason)
            return None

        self.ik_success += 1
        max_delta = distancia_articular(solution, seed)

        rospy.logdebug("TRAC-IK OK | %.3f ms | delta_seed=%.4f rad", elapsed * 1000.0, max_delta)
        return solution

    def _pose_callback(self, msg):
        if not self.has_received_joints:
            rospy.logwarn_throttle(2.0, "Aguardando /joint_states antes de executar IK.")
            return

        pose = self._extract_pose(msg)
        if pose is None:
            return

        self._check_external_drift()

        if self.is_first_ik:
            seed = list(self.q_init)
            rospy.loginfo("TRAC-IK: usando posição física como seed.")
        else:
            seed = list(self.last_q_out)

        if not all_finite(seed):
            rospy.logerr("Seed inválido. Resetando para posição física.")
            seed = list(self.q_init)
            self.is_first_ik = True

        solution = self._solve_ik(pose, seed)

        if solution is None:
            return

        self.last_q_out = list(solution)
        self.is_first_ik = False
        self._publish_joint_command(solution)

    def _autonomous_pose_callback(self, msg):
        if not self.has_received_joints:
            rospy.logwarn_throttle(2.0, "Autonomous IK aguardando /joint_states.")
            return

        pose = self._extract_pose(msg)
        if pose is None:
            return

        seed = list(self.q_init)
        rospy.loginfo("TRAC-IK AUTONOMOUS: usando posição física como seed.")

        solution = self._solve_ik(pose, seed)

        if solution is None:
            rospy.logwarn("Autonomous IK falhou. Nenhum comando autônomo publicado.")
            return

        self.last_q_out = list(solution)
        self.is_first_ik = True

        target_msg = JointState()
        target_msg.header = Header()
        target_msg.header.stamp = rospy.Time.now()
        target_msg.name = JOINT_NAMES + ["finger_joint"]
        target_msg.position = list(solution) + [0.0]

        self.target_joints_pub.publish(target_msg)
        rospy.loginfo("Autonomous target publicado.")

    def _publish_joint_command(self, target):
        if not all_finite(target):
            rospy.logerr("Comando não publicado: target contém NaN/Inf.")
            return

        if len(target) != len(JOINT_NAMES):
            rospy.logerr("Comando não publicado: número de joints inválido.")
            return

        for i, q in enumerate(target):
            if q < JOINT_MIN[i] - 1e-6 or q > JOINT_MAX[i] + 1e-6:
                rospy.logerr(
                    "Comando rejeitado: %s = %.5f fora de [%.5f, %.5f]",
                    JOINT_NAMES[i], q, JOINT_MIN[i], JOINT_MAX[i]
                )
                return

        max_delta = distancia_articular(target, self.q_init)

        # [ROBÔ REAL]: O piso dinâmico evita violações na caixa controladora
        dynamic_duration = max(MIN_COMMAND_DURATION, max_delta / MAX_RAD_PER_SEC)

        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = BASE_LINK
        trajectory.joint_names = list(JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = list(target)
        point.velocities = [0.0] * len(JOINT_NAMES)
        point.accelerations = [0.0] * len(JOINT_NAMES)
        point.time_from_start = rospy.Duration(dynamic_duration)

        trajectory.points.append(point)

        if SEND_TO_HARDWARE:
            self.command_pub.publish(trajectory)
        elif TEST_MODE:
            rospy.loginfo("TEST_MODE: comando calculado, não enviado.")

        rospy.logdebug("Comando publicado | max_delta=%.4f rad | duration=%.3f s", max_delta, dynamic_duration)

    def run_test(self):
        rospy.loginfo("================================================")
        rospy.loginfo(" INICIANDO TESTE TRAC-IK (ROBÔ REAL)")
        rospy.loginfo("================================================")

        if not self.has_received_joints:
            rospy.logwarn("Teste aguardando JointState...")
            while not rospy.is_shutdown() and not self.has_received_joints:
                rospy.sleep(0.1)

        test_pose = PoseStamped()
        test_pose.header.frame_id = BASE_LINK
        test_pose.pose.position.x = 0.40
        test_pose.pose.position.y = 0.00
        test_pose.pose.position.z = 0.40
        test_pose.pose.orientation.x = 0.0
        test_pose.pose.orientation.y = 0.0
        test_pose.pose.orientation.z = 0.0
        test_pose.pose.orientation.w = 1.0

        pose = self._extract_pose(test_pose)
        if pose is None:
            rospy.logerr("Pose de teste inválida.")
            return

        seed = list(self.q_init)
        rospy.loginfo("Seed de teste: %s", str(seed))

        solution = self._solve_ik(pose, seed)

        if solution is None:
            rospy.logerr("TESTE TRAC-IK: FALHOU.")
            return

        rospy.loginfo("TESTE TRAC-IK: SUCESSO.")
        rospy.loginfo("Seed: %s", str(seed))
        rospy.loginfo("Solution: %s", str(solution))
        rospy.loginfo("Tempo IK: %.3f ms", self.last_ik_time * 1000.0)
        rospy.loginfo("Delta máximo: %.4f rad", distancia_articular(solution, seed))


def main():
    rospy.init_node("unified_trac_ik_teleop_solver_real", anonymous=False)

    try:
        solver = UnifiedTRACIKTeleopSolver()
        if TEST_MODE:
            solver.run_test()

        rospy.loginfo("Unified TRAC-IK Teleop Solver rodando no Robô Real.")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("Falha fatal no Unified TRAC-IK Solver: %s", str(exc))
        raise

if __name__ == "__main__":
    main()