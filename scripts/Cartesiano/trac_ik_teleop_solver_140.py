#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
UNIFIED TRAC-IK TELEOP SOLVER
===============================================================================

Arquitetura:

    Unity VR
       |
       |  /unity/target_pose
       v
+-----------------------------+
|       TRAC-IK Solver        |
|                             |
|  - Distance mode            |
|  - Seed físico / temporal   |
|  - Joint limits             |
|  - Anti-unwinding           |
|  - Validação pós-IK         |
|  - Proteção contra NaN/Inf  |
|  - Detecção de drift        |
+-----------------------------+
       |
       | /ur5/planned_trajectory
       v
+-----------------------------+
|     Quintic Planner         |
+-----------------------------+
       |
       v
   Robot Commander
       |
       v
 Gazebo / UR5 real


ESTRATÉGIA DE SEED
------------------

1. Primeiro IK após inicialização:
       seed = posição física atual

2. Durante teleop:
       seed = última solução IK válida

3. Se o robô for movido fisicamente/menu:
       detecta drift > threshold
       seed = posição física novamente

4. Comando autônomo:
       seed = posição física
       depois força reset do teleop


LIMITES
-------

Mesmos limites definidos na implementação KDL de referência:

    shoulder_pan   [-2*pi, +2*pi]
    shoulder_lift  [-2.3, 0.0]
    elbow          [-pi, -0.1]
    wrist_1        [-2*pi, +2*pi]
    wrist_2        [0.1, 3.0]
    wrist_3        [-2*pi, +2*pi]


IMPORTANTE
----------

Este nó não faz detecção de colisão geométrica.
Joint limits e "virtual cast" não substituem collision checking.

O comando final ainda deve passar pelo planner/controle que impõe
velocidade e aceleração máximas no movimento real.
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
SEND_TO_GAZEBO = True

INPUT_AS_QUATERNION = True

BASE_LINK = "base_link"
EE_LINK = "tool0"

POSE_TOPIC = "unity/target_pose"
AUTONOMOUS_POSE_TOPIC = "unity/target_pose_autonomous"

COMMAND_TOPIC = "/ur5/planned_trajectory"

JOINT_STATES_TOPIC = "/ur5/joint_states"

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

# 5 ms é mantido para comparação com a versão anterior.
TRACIK_TIMEOUT = 0.005

# Mantido em 1e-4 para a primeira comparação científica com seu código.
TRACIK_EPSILON = 1e-4

# Distance procura uma solução próxima ao seed.
TRACIK_SOLVE_TYPE = "Distance"


# =============================================================================
# SEGURANÇA / CONTROLE
# =============================================================================

# Se a posição física divergir desta quantidade da última solução,
# consideramos que o robô foi movido externamente.
MENU_DRIFT_THRESHOLD = 0.20  # rad ~= 11.5 graus


# Velocidade máxima utilizada para calcular o tempo do comando intermediário.
MAX_RAD_PER_SEC = 2.0

MIN_COMMAND_DURATION = 0.20   # 0.10 ou 0.05


# Limites máximos gerais do envelope virtual.
DEFAULT_MIN = -2.0 * math.pi
DEFAULT_MAX = +2.0 * math.pi


# =============================================================================
# LIMITES ARTICULARES
# =============================================================================

JOINT_MIN = [
    DEFAULT_MIN,   # shoulder_pan
    -2.3,          # shoulder_lift
    -math.pi,      # elbow
    DEFAULT_MIN,   # wrist_1
    0.1,           # wrist_2
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
    """
    Verifica se um valor é numérico e finito.
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def all_finite(values):
    """
    Verifica uma lista inteira.
    """
    return all(is_finite(v) for v in values)


def circular_difference(target, current):
    """
    Diferença angular mínima:

        target - current

    normalizada em [-pi, +pi].
    """
    return math.atan2(
        math.sin(target - current),
        math.cos(target - current)
    )


def resolver_salto_angular(angulo_alvo, angulo_atual):
    """
    Mantém o movimento pelo caminho angular mais curto.

    Também mantém o resultado dentro de aproximadamente +/- 2 revoluções,
    como na implementação KDL original.
    """

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
    """
    Distância articular máxima considerando a natureza circular das juntas.
    """

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
        rospy.loginfo(" UNIFIED TRAC-IK TELEOP SOLVER")
        rospy.loginfo("================================================")

        # ---------------------------------------------------------------------
        # Estado articular
        # ---------------------------------------------------------------------

        self.has_received_joints = False

        self.q_init = [0.0] * 6

        self.last_q_out = [0.0] * 6

        self.is_first_ik = True

        # ---------------------------------------------------------------------
        # Lock
        # ---------------------------------------------------------------------

        # TRAC-IK/KDL não deve ser chamado concorrentemente através da mesma
        # instância.
        self.ik_lock = threading.Lock()

        # ---------------------------------------------------------------------
        # Estatísticas
        # ---------------------------------------------------------------------

        self.ik_count = 0
        self.ik_success = 0
        self.ik_failure = 0

        self.last_ik_time = 0.0

        # ---------------------------------------------------------------------
        # Inicialização
        # ---------------------------------------------------------------------

        self._initialize_trac_ik()

        self._setup_ros()

        rospy.loginfo("TRAC-IK solver inicializado com sucesso.")


    # =========================================================================
    # TRAC-IK
    # =========================================================================

    def _initialize_trac_ik(self):

        rospy.loginfo("Inicializando TRAC-IK...")

        try:

            # -------------------------------------------------------------
            # Obtém URDF
            # -------------------------------------------------------------

            urdf_string = rospy.get_param("/robot_description")

            if not urdf_string:
                raise RuntimeError(
                    "Parâmetro /robot_description está vazio."
                )

            # -------------------------------------------------------------
            # Modifica URDF para manter a filosofia do "gesso virtual"
            # -------------------------------------------------------------

            modified_urdf = self._inject_joint_limits(
                urdf_string
            )

            # -------------------------------------------------------------
            # Cria uma única instância TRAC-IK
            # -------------------------------------------------------------

            self.ik_solver = IK(
                BASE_LINK,
                EE_LINK,
                urdf_string=modified_urdf,
                timeout=TRACIK_TIMEOUT,
                epsilon=TRACIK_EPSILON,
                solve_type=TRACIK_SOLVE_TYPE
            )

            # -------------------------------------------------------------
            # Verifica cadeia
            # -------------------------------------------------------------

            rospy.loginfo(
                "TRAC-IK encontrou %d joints.",
                self.ik_solver.number_of_joints
            )

            rospy.loginfo(
                "TRAC-IK joint names: %s",
                str(self.ik_solver.joint_names)
            )

            if self.ik_solver.number_of_joints != len(JOINT_NAMES):

                raise RuntimeError(
                    "Número de joints incompatível: "
                    "TRAC-IK=%d, esperado=%d"
                    % (
                        self.ik_solver.number_of_joints,
                        len(JOINT_NAMES)
                    )
                )

            # -------------------------------------------------------------
            # Segurança crítica:
            # a ordem retornada pelo TRAC-IK precisa ser exatamente
            # a ordem utilizada pelo restante do pipeline.
            # -------------------------------------------------------------

            if list(self.ik_solver.joint_names) != JOINT_NAMES:

                raise RuntimeError(
                    "ORDEM DOS JOINTS INCOMPATÍVEL!\n"
                    "Esperado: %s\n"
                    "TRAC-IK:  %s"
                    % (
                        JOINT_NAMES,
                        self.ik_solver.joint_names
                    )
                )

            # -------------------------------------------------------------
            # Segunda camada:
            # força explicitamente os limites dentro do TRAC-IK.
            #
            # Isso evita depender somente do XML modificado.
            # -------------------------------------------------------------

            self.ik_solver.set_joint_limits(
                JOINT_MIN,
                JOINT_MAX
            )

            lower, upper = self.ik_solver.get_joint_limits()

            rospy.loginfo(
                "TRAC-IK lower limits: %s",
                str(lower)
            )

            rospy.loginfo(
                "TRAC-IK upper limits: %s",
                str(upper)
            )

            # -------------------------------------------------------------
            # Verificação final dos limites
            # -------------------------------------------------------------

            for i in range(len(JOINT_NAMES)):

                if abs(lower[i] - JOINT_MIN[i]) > 1e-6:
                    raise RuntimeError(
                        "Lower limit incorreto no joint %s."
                        % JOINT_NAMES[i]
                    )

                if abs(upper[i] - JOINT_MAX[i]) > 1e-6:
                    raise RuntimeError(
                        "Upper limit incorreto no joint %s."
                        % JOINT_NAMES[i]
                    )

            rospy.loginfo(
                "TRAC-IK configurado: "
                "timeout=%.4f s | epsilon=%.1e | solve_type=%s",
                TRACIK_TIMEOUT,
                TRACIK_EPSILON,
                TRACIK_SOLVE_TYPE
            )

        except Exception as exc:

            rospy.logerr(
                "Falha crítica ao inicializar TRAC-IK: %s",
                str(exc)
            )

            raise


    # =========================================================================
    # INJEÇÃO DE LIMITES NO URDF
    # =========================================================================

    def _inject_joint_limits(self, urdf_string):

        try:

            root = ET.fromstring(urdf_string)

        except Exception as exc:

            raise RuntimeError(
                "Não foi possível interpretar /robot_description: %s"
                % str(exc)
            )

        joint_map = {}

        for joint in root.findall("joint"):

            name = joint.get("name")

            if name in JOINT_NAMES:
                joint_map[name] = joint

        # ---------------------------------------------------------------------
        # Verifica se todos os joints existem
        # ---------------------------------------------------------------------

        missing = [
            name
            for name in JOINT_NAMES
            if name not in joint_map
        ]

        if missing:

            raise RuntimeError(
                "Joints ausentes no URDF: %s"
                % str(missing)
            )

        # ---------------------------------------------------------------------
        # Modifica TODOS os limites para garantir equivalência com KDL.
        #
        # Isso é importante:
        # não queremos que um limite original do URDF torne o experimento
        # KDL x TRAC-IK diferente sem percebermos.
        # ---------------------------------------------------------------------

        for i, joint_name in enumerate(JOINT_NAMES):

            joint = joint_map[joint_name]

            limit = joint.find("limit")

            if limit is None:

                limit = ET.SubElement(
                    joint,
                    "limit"
                )

            limit.set(
                "lower",
                str(JOINT_MIN[i])
            )

            limit.set(
                "upper",
                str(JOINT_MAX[i])
            )

        modified_urdf = ET.tostring(
            root,
            encoding="unicode"
        )

        rospy.loginfo(
            "Limites virtuais TRAC-IK injetados no URDF."
        )

        return modified_urdf


    # =========================================================================
    # ROS
    # =========================================================================

    def _setup_ros(self):

        # ---------------------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------------------

        self.command_pub = rospy.Publisher(
            COMMAND_TOPIC,
            JointTrajectory,
            queue_size=1
        )

        self.target_joints_pub = rospy.Publisher(
            TARGET_JOINTS_TOPIC,
            JointState,
            queue_size=1
        )

        # ---------------------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------------------

        rospy.Subscriber(
            JOINT_STATES_TOPIC,
            JointState,
            self._joint_state_callback,
            queue_size=1
        )

        rospy.Subscriber(
            POSE_TOPIC,
            PoseStamped,
            self._pose_callback,
            queue_size=1
        )

        rospy.Subscriber(
            AUTONOMOUS_POSE_TOPIC,
            PoseStamped,
            self._autonomous_pose_callback,
            queue_size=1
        )

        rospy.loginfo(
            "Subscriber: %s",
            JOINT_STATES_TOPIC
        )

        rospy.loginfo(
            "Subscriber: %s",
            POSE_TOPIC
        )

        rospy.loginfo(
            "Subscriber: %s",
            AUTONOMOUS_POSE_TOPIC
        )

        rospy.loginfo(
            "Publisher: %s",
            COMMAND_TOPIC
        )


    # =========================================================================
    # JOINT STATES
    # =========================================================================

    def _joint_state_callback(self, msg):

        q_new = [None] * len(JOINT_NAMES)

        # ---------------------------------------------------------------------
        # Mapeia por nome.
        #
        # Nunca assumir que a ordem do JointState é a mesma da UR5.
        # ---------------------------------------------------------------------

        for expected_index, expected_name in enumerate(JOINT_NAMES):

            found = False

            for i, name in enumerate(msg.name):

                if name == expected_name:

                    if i < len(msg.position):

                        q_new[expected_index] = (
                            float(msg.position[i])
                        )

                        found = True

                    break

            if not found:
                break

        # ---------------------------------------------------------------------
        # Só aceita o estado se TODOS os joints estiverem presentes.
        # ---------------------------------------------------------------------

        if any(q is None for q in q_new):

            return

        if not all_finite(q_new):

            rospy.logwarn_throttle(
                2.0,
                "JointState contém valor NaN/Inf."
            )

            return

        self.q_init = q_new
        self.has_received_joints = True


    # =========================================================================
    # POSE NORMALIZATION
    # =========================================================================

    def _extract_pose(self, msg):

        p = msg.pose.position
        q = msg.pose.orientation

        values = [
            p.x,
            p.y,
            p.z,
            q.x,
            q.y,
            q.z,
            q.w
        ]

        if not all_finite(values):

            rospy.logwarn(
                "Pose descartada: contém NaN/Inf."
            )

            return None

        # ---------------------------------------------------------------------
        # Normalização do quaternion
        # ---------------------------------------------------------------------

        q_norm = math.sqrt(
            q.x * q.x +
            q.y * q.y +
            q.z * q.z +
            q.w * q.w
        )

        if not math.isfinite(q_norm):

            rospy.logwarn(
                "Pose descartada: norma do quaternion inválida."
            )

            return None

        if q_norm < 1e-9:

            rospy.logwarn(
                "Pose descartada: quaternion praticamente nulo."
            )

            return None

        qx = q.x / q_norm
        qy = q.y / q_norm
        qz = q.z / q_norm
        qw = q.w / q_norm

        return (
            float(p.x),
            float(p.y),
            float(p.z),
            float(qx),
            float(qy),
            float(qz),
            float(qw)
        )


    # =========================================================================
    # MENU / RESET DETECTOR
    # =========================================================================

    def _check_external_drift(self):

        if self.is_first_ik:
            return

        drift = distancia_articular(
            self.q_init,
            self.last_q_out
        )

        if drift > MENU_DRIFT_THRESHOLD:

            rospy.logwarn(
                "DRIFT EXTERNO detectado: %.3f rad "
                "(threshold %.3f). "
                "Resetando seed para posição física.",
                drift,
                MENU_DRIFT_THRESHOLD
            )

            self.is_first_ik = True


    # =========================================================================
    # VALIDAÇÃO DA SOLUÇÃO
    # =========================================================================

    def _validate_solution(self, solution, seed):

        # ---------------------------------------------------------------------
        # Estrutura
        # ---------------------------------------------------------------------

        if solution is None:

            return False, "TRAC-IK retornou None"

        if len(solution) != len(JOINT_NAMES):

            return (
                False,
                "número de joints incorreto"
            )

        # ---------------------------------------------------------------------
        # Finite
        # ---------------------------------------------------------------------

        if not all_finite(solution):

            return (
                False,
                "solução contém NaN/Inf"
            )

        # ---------------------------------------------------------------------
        # Limites articulares
        # ---------------------------------------------------------------------

        for i, q in enumerate(solution):

            if q < JOINT_MIN[i] - 1e-6:

                return (
                    False,
                    "joint %s abaixo do limite: %.6f < %.6f"
                    % (
                        JOINT_NAMES[i],
                        q,
                        JOINT_MIN[i]
                    )
                )

            if q > JOINT_MAX[i] + 1e-6:

                return (
                    False,
                    "joint %s acima do limite: %.6f > %.6f"
                    % (
                        JOINT_NAMES[i],
                        q,
                        JOINT_MAX[i]
                    )
                )

        # ---------------------------------------------------------------------
        # Salto angular
        # ---------------------------------------------------------------------

        for i in range(len(solution)):

            delta = abs(
                circular_difference(
                    solution[i],
                    seed[i]
                )
            )

            # Mais de 180 graus de diferença é considerado suspeito.
            #
            # TRAC-IK Distance normalmente já favorece soluções próximas
            # ao seed, mas esta camada impede um resultado claramente
            # incompatível com a estratégia de teleop.
            if delta > math.pi + 1e-6:

                return (
                    False,
                    "salto angular suspeito no joint %s: %.3f rad"
                    % (
                        JOINT_NAMES[i],
                        delta
                    )
                )

        return True, "OK"


    # =========================================================================
    # SOLUÇÃO IK
    # =========================================================================

    def _solve_ik(self, pose, seed):

        px, py, pz, qx, qy, qz, qw = pose

        start = time.monotonic()

        try:

            # -----------------------------------------------------------------
            # Lock:
            # apenas uma chamada ao TRAC-IK por vez.
            # -----------------------------------------------------------------

            with self.ik_lock:

                solution = self.ik_solver.get_ik(
                    list(seed),
                    px,
                    py,
                    pz,
                    qx,
                    qy,
                    qz,
                    qw
                )

        except Exception as exc:

            elapsed = time.monotonic() - start

            self.last_ik_time = elapsed

            self.ik_failure += 1

            rospy.logerr(
                "Exceção no TRAC-IK após %.3f ms: %s",
                elapsed * 1000.0,
                str(exc)
            )

            return None

        elapsed = time.monotonic() - start

        self.last_ik_time = elapsed
        self.ik_count += 1

        # ---------------------------------------------------------------------
        # Falha do solver
        # ---------------------------------------------------------------------

        if solution is None:

            self.ik_failure += 1

            rospy.logwarn_throttle(
                1.0,
                "TRAC-IK não encontrou solução. "
                "tempo=%.3f ms",
                elapsed * 1000.0
            )

            return None

        # ---------------------------------------------------------------------
        # Converte para lista
        # ---------------------------------------------------------------------

        solution = [
            float(q)
            for q in solution
        ]

        # ---------------------------------------------------------------------
        # Anti-unwinding
        # ---------------------------------------------------------------------

        corrected_solution = []

        for i in range(len(solution)):

            q_corrected = resolver_salto_angular(
                solution[i],
                seed[i]
            )

            corrected_solution.append(
                q_corrected
            )

        solution = corrected_solution

        # ---------------------------------------------------------------------
        # Validação final
        # ---------------------------------------------------------------------

        valid, reason = self._validate_solution(
            solution,
            seed
        )

        if not valid:

            self.ik_failure += 1

            rospy.logwarn(
                "Solução TRAC-IK rejeitada: %s",
                reason
            )

            return None

        # ---------------------------------------------------------------------
        # Métrica
        # ---------------------------------------------------------------------

        self.ik_success += 1

        max_delta = distancia_articular(
            solution,
            seed
        )

        rospy.logdebug(
            "TRAC-IK OK | %.3f ms | "
            "delta_seed=%.4f rad",
            elapsed * 1000.0,
            max_delta
        )

        return solution


    # =========================================================================
    # TELEOP CALLBACK
    # =========================================================================

    def _pose_callback(self, msg):

        if not self.has_received_joints:

            rospy.logwarn_throttle(
                2.0,
                "Aguardando /joint_states antes de executar IK."
            )

            return

        pose = self._extract_pose(msg)

        if pose is None:
            return

        # ---------------------------------------------------------------------
        # Detecta movimentação externa / menu
        # ---------------------------------------------------------------------

        self._check_external_drift()

        # ---------------------------------------------------------------------
        # Seleção do seed
        # ---------------------------------------------------------------------

        if self.is_first_ik:

            seed = list(self.q_init)

            rospy.loginfo(
                "TRAC-IK: usando posição física como seed."
            )

        else:

            seed = list(self.last_q_out)

        # ---------------------------------------------------------------------
        # Segurança do seed
        # ---------------------------------------------------------------------

        if not all_finite(seed):

            rospy.logerr(
                "Seed inválido. Resetando para posição física."
            )

            seed = list(self.q_init)
            self.is_first_ik = True

        # ---------------------------------------------------------------------
        # IK
        # ---------------------------------------------------------------------

        solution = self._solve_ik(
            pose,
            seed
        )

        if solution is None:

            # Não publica nada.
            # Mantém o último comando válido.
            return

        # ---------------------------------------------------------------------
        # Atualiza estado SOMENTE após solução validada.
        # ---------------------------------------------------------------------

        self.last_q_out = list(solution)

        self.is_first_ik = False

        # ---------------------------------------------------------------------
        # Publica
        # ---------------------------------------------------------------------

        self._publish_joint_command(
            solution
        )


    # =========================================================================
    # AUTONOMOUS CALLBACK
    # =========================================================================

    def _autonomous_pose_callback(self, msg):

        if not self.has_received_joints:

            rospy.logwarn_throttle(
                2.0,
                "Autonomous IK aguardando /joint_states."
            )

            return

        pose = self._extract_pose(msg)

        if pose is None:
            return

        # ---------------------------------------------------------------------
        # AUTONOMOUS SEMPRE começa da posição física.
        #
        # Isso evita herdar uma solução virtual anterior.
        # ---------------------------------------------------------------------

        seed = list(self.q_init)

        rospy.loginfo(
            "TRAC-IK AUTONOMOUS: usando posição física como seed."
        )

        solution = self._solve_ik(
            pose,
            seed
        )

        if solution is None:

            rospy.logwarn(
                "Autonomous IK falhou. "
                "Nenhum comando autônomo publicado."
            )

            return

        # ---------------------------------------------------------------------
        # Atualiza última solução
        # ---------------------------------------------------------------------

        self.last_q_out = list(solution)

        # ---------------------------------------------------------------------
        # Depois de uma ação autônoma, o próximo movimento VR deve novamente
        # começar da realidade física.
        # ---------------------------------------------------------------------

        self.is_first_ik = True

        # ---------------------------------------------------------------------
        # Publica alvo autônomo
        # ---------------------------------------------------------------------

        target_msg = JointState()

        target_msg.header = Header()
        target_msg.header.stamp = rospy.Time.now()

        target_msg.name = JOINT_NAMES + [
            "finger_joint"
        ]

        target_msg.position = list(solution) + [
            0.0
        ]

        self.target_joints_pub.publish(
            target_msg
        )

        rospy.loginfo(
            "Autonomous target publicado."
        )


    # =========================================================================
    # PUBLICAÇÃO
    # =========================================================================

    def _publish_joint_command(self, target):

        if not all_finite(target):

            rospy.logerr(
                "Comando não publicado: target contém NaN/Inf."
            )

            return

        if len(target) != len(JOINT_NAMES):

            rospy.logerr(
                "Comando não publicado: número de joints inválido."
            )

            return

        # ---------------------------------------------------------------------
        # Validação adicional dos limites
        # ---------------------------------------------------------------------

        for i, q in enumerate(target):

            if q < JOINT_MIN[i] - 1e-6 or q > JOINT_MAX[i] + 1e-6:

                rospy.logerr(
                    "Comando rejeitado: %s = %.5f fora de [%.5f, %.5f]",
                    JOINT_NAMES[i],
                    q,
                    JOINT_MIN[i],
                    JOINT_MAX[i]
                )

                return

        # ---------------------------------------------------------------------
        # Distância em relação à posição física.
        #
        # IMPORTANTE:
        # q_init representa a realidade física mais recente.
        # ---------------------------------------------------------------------

        max_delta = distancia_articular(
            target,
            self.q_init
        )

        # ---------------------------------------------------------------------
        # Tempo dinâmico.
        #
        # Isto mantém a metodologia do KDL original.
        # ---------------------------------------------------------------------

        dynamic_duration = max(
            MIN_COMMAND_DURATION,
            max_delta / MAX_RAD_PER_SEC
        )

        # ---------------------------------------------------------------------
        # Mensagem
        # ---------------------------------------------------------------------

        trajectory = JointTrajectory()

        trajectory.header.stamp = rospy.Time.now()

        trajectory.header.frame_id = BASE_LINK

        trajectory.joint_names = list(
            JOINT_NAMES
        )

        point = JointTrajectoryPoint()

        point.positions = list(target)

        # O planner downstream poderá recalcular a trajetória.
        # Mantemos explicitamente velocidade/aceleração zero no ponto-alvo.
        point.velocities = [0.0] * len(JOINT_NAMES)
        point.accelerations = [0.0] * len(JOINT_NAMES)

        point.time_from_start = rospy.Duration(
            dynamic_duration
        )

        trajectory.points.append(
            point
        )

        # ---------------------------------------------------------------------
        # Publicação
        # ---------------------------------------------------------------------

        if SEND_TO_GAZEBO:

            self.command_pub.publish(
                trajectory
            )

        elif TEST_MODE:

            rospy.loginfo(
                "TEST_MODE: comando calculado, não enviado."
            )

        rospy.logdebug(
            "Comando publicado | "
            "max_delta=%.4f rad | duration=%.3f s",
            max_delta,
            dynamic_duration
        )


    # =========================================================================
    # TESTE
    # =========================================================================

    def run_test(self):

        rospy.loginfo(
            "================================================"
        )

        rospy.loginfo(
            " INICIANDO TESTE TRAC-IK"
        )

        rospy.loginfo(
            "================================================"
        )

        if not self.has_received_joints:

            rospy.logwarn(
                "Teste aguardando JointState..."
            )

            while (
                not rospy.is_shutdown()
                and not self.has_received_joints
            ):

                rospy.sleep(0.1)

        # ---------------------------------------------------------------------
        # Pose de teste
        # ---------------------------------------------------------------------

        test_pose = PoseStamped()

        test_pose.header.frame_id = BASE_LINK

        test_pose.pose.position.x = 0.40
        test_pose.pose.position.y = 0.00
        test_pose.pose.position.z = 0.40

        test_pose.pose.orientation.x = 0.0
        test_pose.pose.orientation.y = 0.0
        test_pose.pose.orientation.z = 0.0
        test_pose.pose.orientation.w = 1.0

        pose = self._extract_pose(
            test_pose
        )

        if pose is None:

            rospy.logerr(
                "Pose de teste inválida."
            )

            return

        # ---------------------------------------------------------------------
        # Teste a partir da realidade
        # ---------------------------------------------------------------------

        seed = list(
            self.q_init
        )

        rospy.loginfo(
            "Seed de teste: %s",
            str(seed)
        )

        solution = self._solve_ik(
            pose,
            seed
        )

        if solution is None:

            rospy.logerr(
                "TESTE TRAC-IK: FALHOU."
            )

            return

        rospy.loginfo(
            "TESTE TRAC-IK: SUCESSO."
        )

        rospy.loginfo(
            "Seed: %s",
            str(seed)
        )

        rospy.loginfo(
            "Solution: %s",
            str(solution)
        )

        rospy.loginfo(
            "Tempo IK: %.3f ms",
            self.last_ik_time * 1000.0
        )

        rospy.loginfo(
            "Delta máximo: %.4f rad",
            distancia_articular(
                solution,
                seed
            )
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    rospy.init_node(
        "unified_trac_ik_teleop_solver",
        anonymous=False
    )

    try:

        solver = UnifiedTRACIKTeleopSolver()

        if TEST_MODE:

            solver.run_test()

        rospy.loginfo(
            "Unified TRAC-IK Teleop Solver rodando."
        )

        rospy.spin()

    except rospy.ROSInterruptException:

        pass

    except Exception as exc:

        rospy.logfatal(
            "Falha fatal no Unified TRAC-IK Solver: %s",
            str(exc)
        )

        raise


if __name__ == "__main__":

    main()
