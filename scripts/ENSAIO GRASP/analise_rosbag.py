#!/usr/bin/env python3

import os
import csv
import math
import rosbag
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_BAG = 'ensaio_grasping_0.bag'

PASTA_RESULTADOS = 'resultados_ensaio'

TOPIC_JOINT_STATES = '/ur5/joint_states'
TOPIC_COMMAND = '/ur5/eff_joint_traj_controller/command'
TOPIC_POINTS = '/points_fixed_to_base'

JUNTAS = [
    'Shoulder Pan',
    'Shoulder Lift',
    'Elbow',
    'Wrist 1',
    'Wrist 2',
    'Wrist 3'
]

# ------------------------------------------------------------
# Parâmetros de processamento
# ------------------------------------------------------------

TARGET_HZ = 50.0

# Filtro Savitzky-Golay
SG_WINDOW = 21
SG_POLYORDER = 3

# ------------------------------------------------------------
# Detecção de movimento
# ------------------------------------------------------------

# Limiar mínimo de velocidade articular agregada.
#
# O algoritmo também utiliza um limiar estatístico calculado
# a partir do nível de atividade da aquisição.
#
# Se necessário, este valor pode ser ajustado.
MIN_MOVE_THRESHOLD = 0.08       # rad/s

# Limiar para considerar que o movimento terminou.
# Utiliza histerese para evitar liga/desliga excessivo.
STOP_THRESHOLD = 0.05           # rad/s

# Duração mínima de um segmento para ser considerado movimento
MIN_MOVEMENT_DURATION = 0.30    # s

# Pequenos intervalos entre movimentos podem ser considerados
# parte do mesmo segmento.
MAX_MERGE_GAP = 0.50             # s

# ------------------------------------------------------------
# Análise das transições
# ------------------------------------------------------------

# Janela usada para calcular velocidade imediatamente antes
# e depois da transição.
TRANSITION_WINDOW = 0.15         # s

# ------------------------------------------------------------
# Estatísticas
# ------------------------------------------------------------

PERCENTILES = [5, 50, 95]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def rms(x):
    """Root Mean Square."""
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.nan

    return np.sqrt(np.mean(x ** 2))


def timestamp_to_sec(stamp):
    """Converte rospy.Time para segundos."""
    return stamp.secs + stamp.nsecs * 1e-9


def safe_savgol(data, window=21, polyorder=3):
    """
    Aplica Savitzky-Golay somente quando existem amostras
    suficientes.
    """

    data = np.asarray(data)

    if len(data) < polyorder + 2:
        return data.copy()

    # A janela precisa ser ímpar
    if window % 2 == 0:
        window += 1

    # Não pode ser maior que o número de amostras
    if window > len(data):
        window = len(data) if len(data) % 2 == 1 else len(data) - 1

    if window <= polyorder:
        return data.copy()

    return savgol_filter(
        data,
        window_length=window,
        polyorder=polyorder,
        axis=0
    )


def percentile_abs(data, percentile):
    """Percentil do valor absoluto."""
    data = np.asarray(data)

    if len(data) == 0:
        return np.nan

    return np.percentile(np.abs(data), percentile)


def ensure_output_folder():
    if not os.path.exists(PASTA_RESULTADOS):
        os.makedirs(PASTA_RESULTADOS)


def save_csv(filename, header, rows):

    path = os.path.join(PASTA_RESULTADOS, filename)

    with open(path, 'w', newline='') as f:

        writer = csv.writer(f)

        writer.writerow(header)

        for row in rows:
            writer.writerow(row)

    return path


# ============================================================
# DETECÇÃO DOS SEGMENTOS DE MOVIMENTO
# ============================================================

def detect_movement_segments(times, velocities):

    """
    Detecta segmentos físicos de movimento a partir da velocidade
    observada em /joint_states.

    A velocidade utilizada é a norma RMS das seis juntas:

        v_norm = sqrt(mean(v_j^2))

    É utilizada histerese entre MOVE_THRESHOLD e STOP_THRESHOLD.
    """

    times = np.asarray(times)
    velocities = np.asarray(velocities)

    # --------------------------------------------------------
    # Velocidade agregada do manipulador
    # --------------------------------------------------------

    speed = np.sqrt(np.mean(velocities ** 2, axis=1))

    # Suavização leve apenas para detecção
    speed_smooth = safe_savgol(
        speed,
        window=SG_WINDOW,
        polyorder=SG_POLYORDER
    )

    # --------------------------------------------------------
    # Estimativa simples do nível de atividade
    # --------------------------------------------------------

    baseline = np.median(speed_smooth)

    mad = np.median(
        np.abs(speed_smooth - baseline)
    )

    robust_noise = 1.4826 * mad

    statistical_threshold = (
        baseline + 6.0 * robust_noise
    )

    move_threshold = max(
        MIN_MOVE_THRESHOLD,
        statistical_threshold
    )

    stop_threshold = min(
        STOP_THRESHOLD,
        move_threshold * 0.8
    )

    print("\nParâmetros de detecção de movimento:")
    print(f"  Velocidade mediana           : {baseline:.5f} rad/s")
    print(f"  MAD robusto                  : {robust_noise:.5f} rad/s")
    print(f"  Limiar estatístico           : {statistical_threshold:.5f} rad/s")
    print(f"  Limiar de início             : {move_threshold:.5f} rad/s")
    print(f"  Limiar de parada             : {stop_threshold:.5f} rad/s")

    # --------------------------------------------------------
    # Máquina de estados
    # --------------------------------------------------------

    segments = []

    in_motion = False
    start_idx = None

    for i in range(len(times)):

        current_speed = speed_smooth[i]

        if not in_motion:

            if current_speed >= move_threshold:

                in_motion = True
                start_idx = i

        else:

            if current_speed <= stop_threshold:

                end_idx = i

                duration = (
                    times[end_idx] -
                    times[start_idx]
                )

                if duration >= MIN_MOVEMENT_DURATION:

                    segments.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start': times[start_idx],
                        'end': times[end_idx],
                        'duration': duration
                    })

                in_motion = False
                start_idx = None

    # Caso o movimento continue até o final
    if in_motion and start_idx is not None:

        end_idx = len(times) - 1

        duration = (
            times[end_idx] -
            times[start_idx]
        )

        if duration >= MIN_MOVEMENT_DURATION:

            segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start': times[start_idx],
                'end': times[end_idx],
                'duration': duration
            })

    # --------------------------------------------------------
    # Merge de segmentos próximos
    # --------------------------------------------------------

    merged = []

    for segment in segments:

        if len(merged) == 0:

            merged.append(segment)

        else:

            previous = merged[-1]

            gap = (
                segment['start'] -
                previous['end']
            )

            if gap <= MAX_MERGE_GAP:

                previous['end_idx'] = segment['end_idx']
                previous['end'] = segment['end']
                previous['duration'] = (
                    previous['end'] -
                    previous['start']
                )

            else:

                merged.append(segment)

    return merged, speed, speed_smooth


# ============================================================
# REAMOSTRAGEM
# ============================================================

def resample_segment(times, velocities, start, end):

    mask = (
        (times >= start) &
        (times <= end)
    )

    t_seg = times[mask]
    v_seg = velocities[mask]

    if len(t_seg) < 5:
        return None, None

    t_uniform = np.arange(
        t_seg[0],
        t_seg[-1],
        1.0 / TARGET_HZ
    )

    if len(t_uniform) < 5:
        return None, None

    v_uniform = np.zeros(
        (len(t_uniform), 6)
    )

    for j in range(6):

        v_uniform[:, j] = np.interp(
            t_uniform,
            t_seg,
            v_seg[:, j]
        )

    return t_uniform, v_uniform


# ============================================================
# MÉTRICAS CINEMÁTICAS
# ============================================================

def calculate_segment_metrics(
    times,
    velocities,
    segments
):

    """
    Calcula velocidade, aceleração e jerk exclusivamente
    nos segmentos físicos de movimento.
    """

    all_velocity = []
    all_acceleration = []
    all_jerk = []

    segment_results = []

    for idx, segment in enumerate(segments):

        t_uniform, v_uniform = resample_segment(
            times,
            velocities,
            segment['start'],
            segment['end']
        )

        if t_uniform is None:
            continue

        # ----------------------------------------------------
        # Filtragem
        # ----------------------------------------------------

        v_filtered = safe_savgol(
            v_uniform,
            SG_WINDOW,
            SG_POLYORDER
        )

        dt = 1.0 / TARGET_HZ

        # ----------------------------------------------------
        # Aceleração
        # ----------------------------------------------------

        acceleration = np.gradient(
            v_filtered,
            dt,
            axis=0
        )

        # ----------------------------------------------------
        # Jerk
        # ----------------------------------------------------

        jerk = np.gradient(
            acceleration,
            dt,
            axis=0
        )

        all_velocity.append(v_filtered)
        all_acceleration.append(acceleration)
        all_jerk.append(jerk)

        # ----------------------------------------------------
        # Métricas do segmento
        # ----------------------------------------------------

        segment_results.append({
            'segment': idx + 1,
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment['duration'],
            'velocity_rms': [
                rms(v_filtered[:, j])
                for j in range(6)
            ],
            'acceleration_rms': [
                rms(acceleration[:, j])
                for j in range(6)
            ],
            'jerk_rms': [
                rms(jerk[:, j])
                for j in range(6)
            ]
        })

    if len(all_velocity) == 0:

        return None

    # Concatenação dos segmentos
    velocity_all = np.vstack(all_velocity)
    acceleration_all = np.vstack(all_acceleration)
    jerk_all = np.vstack(all_jerk)

    metrics = {

        'velocity_rms': np.array([
            rms(velocity_all[:, j])
            for j in range(6)
        ]),

        'velocity_peak': np.array([
            np.max(np.abs(velocity_all[:, j]))
            for j in range(6)
        ]),

        'velocity_p95': np.array([
            percentile_abs(
                velocity_all[:, j],
                95
            )
            for j in range(6)
        ]),

        'acceleration_rms': np.array([
            rms(acceleration_all[:, j])
            for j in range(6)
        ]),

        'acceleration_peak': np.array([
            np.max(np.abs(acceleration_all[:, j]))
            for j in range(6)
        ]),

        'jerk_rms': np.array([
            rms(jerk_all[:, j])
            for j in range(6)
        ]),

        'jerk_peak': np.array([
            np.max(np.abs(jerk_all[:, j]))
            for j in range(6)
        ]),

        'jerk_p95': np.array([
            percentile_abs(
                jerk_all[:, j],
                95
            )
            for j in range(6)
        ])
    }

    return {
        'metrics': metrics,
        'segment_results': segment_results,
        'velocity': velocity_all,
        'acceleration': acceleration_all,
        'jerk': jerk_all
    }


# ============================================================
# ANÁLISE DAS TRANSIÇÕES FÍSICAS
# ============================================================

def analyze_transitions(
    times,
    velocities,
    segments
):

    """
    Analisa exclusivamente as transições entre segmentos
    físicos de movimento.

    Para cada transição:

        Δv = |v_after - v_before|

    onde v_before e v_after são médias calculadas em pequenas
    janelas temporais imediatamente antes e depois da transição.
    """

    transition_results = []

    if len(segments) < 2:

        return transition_results

    for i in range(len(segments) - 1):

        previous = segments[i]
        next_segment = segments[i + 1]

        transition_time = (
            previous['end'] +
            next_segment['start']
        ) / 2.0

        # --------------------------------------------
        # Janela antes
        # --------------------------------------------

        before_start = max(
            previous['start'],
            previous['end'] - TRANSITION_WINDOW
        )

        before_mask = (
            (times >= before_start) &
            (times <= previous['end'])
        )

        # --------------------------------------------
        # Janela depois
        # --------------------------------------------

        after_end = min(
            next_segment['end'],
            next_segment['start'] + TRANSITION_WINDOW
        )

        after_mask = (
            (times >= next_segment['start']) &
            (times <= after_end)
        )

        if (
            np.sum(before_mask) < 2 or
            np.sum(after_mask) < 2
        ):
            continue

        v_before = np.mean(
            velocities[before_mask],
            axis=0
        )

        v_after = np.mean(
            velocities[after_mask],
            axis=0
        )

        delta_v = np.abs(
            v_after - v_before
        )

        transition_results.append({

            'transition': i + 1,

            'time': transition_time,

            'segment_before': i + 1,

            'segment_after': i + 2,

            'delta_v': delta_v,

            'v_before': v_before,

            'v_after': v_after
        })

    return transition_results


# ============================================================
# ANÁLISE DOS COMANDOS
# ============================================================

def analyze_commands(commands):

    durations = [
        c['duration']
        for c in commands
        if np.isfinite(c['duration'])
    ]

    if len(durations) == 0:

        return

    print("\n" + "=" * 80)
    print("6. INTERFACE DE COMANDOS DO CONTROLADOR")
    print("=" * 80)

    print(
        f"Mensagens /command recebidas: "
        f"{len(commands)}"
    )

    print(
        f"Duração média declarada: "
        f"{np.mean(durations):.4f} s"
    )

    print(
        f"Desvio padrão: "
        f"{np.std(durations):.4f} s"
    )

    print(
        f"Mínimo: "
        f"{np.min(durations):.4f} s"
    )

    print(
        f"Máximo: "
        f"{np.max(durations):.4f} s"
    )

    print(
        "\nOBSERVAÇÃO:"
    )

    print(
        "As mensagens /command não são "
        "interpretadas como trajetórias físicas independentes."
    )


# ============================================================
# GRÁFICO: PERFIS DE VELOCIDADE
# ============================================================

def plot_velocity_profiles(
    time_axis,
    velocity
):

    plt.figure(figsize=(13, 7))

    for j in range(6):

        plt.plot(
            time_axis,
            velocity[:, j],
            label=JUNTAS[j],
            linewidth=1.2
        )

    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade articular (rad/s)')
    plt.title(
        'Perfis de Velocidade Durante o Movimento Autônomo'
    )

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        PASTA_RESULTADOS,
        'joint_velocity_profiles_autonomy.png'
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    return path


# ============================================================
# GRÁFICO: JERK RMS
# ============================================================

def plot_jerk_rms(metrics):

    values = metrics['jerk_rms']

    plt.figure(figsize=(11, 6))

    plt.bar(
        JUNTAS,
        values
    )

    plt.ylabel('Jerk RMS (rad/s³)')
    plt.xlabel('Junta')
    plt.title('Jerk RMS por Junta')

    plt.xticks(
        rotation=30,
        ha='right'
    )

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.5
    )

    plt.tight_layout()

    path = os.path.join(
        PASTA_RESULTADOS,
        'jerk_rms_by_joint_autonomy.png'
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    return path


# ============================================================
# GRÁFICO: VELOCIDADE RMS
# ============================================================

def plot_velocity_rms(metrics):

    values = metrics['velocity_rms']

    plt.figure(figsize=(11, 6))

    plt.bar(
        JUNTAS,
        values
    )

    plt.ylabel('Velocidade RMS (rad/s)')
    plt.xlabel('Junta')
    plt.title('Velocidade RMS por Junta')

    plt.xticks(
        rotation=30,
        ha='right'
    )

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.5
    )

    plt.tight_layout()

    path = os.path.join(
        PASTA_RESULTADOS,
        'velocity_rms_by_joint_autonomy.png'
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    return path


# ============================================================
# GRÁFICO: TRANSIÇÕES
# ============================================================

def plot_transition_analysis(
    transition_results
):

    if len(transition_results) == 0:

        return None

    values = np.array([
        r['delta_v']
        for r in transition_results
    ])

    plt.figure(figsize=(13, 7))

    for j in range(6):

        plt.plot(
            np.arange(1, len(values) + 1),
            values[:, j],
            marker='o',
            markersize=4,
            label=JUNTAS[j]
        )

    plt.xlabel('Transição entre segmentos')
    plt.ylabel(r'$|\Delta v|$ (rad/s)')
    plt.title(
        'Variação de Velocidade nas Transições Físicas'
    )

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        PASTA_RESULTADOS,
        'velocity_changes_movement_segments.png'
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    return path


# ============================================================
# GRÁFICO: DURAÇÃO DOS SEGMENTOS
# ============================================================

def plot_segment_durations(segments):

    if len(segments) == 0:

        return None

    durations = [
        s['duration']
        for s in segments
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        np.arange(1, len(durations) + 1),
        durations
    )

    plt.xlabel('Segmento de movimento')
    plt.ylabel('Duração (s)')
    plt.title(
        'Duração dos Segmentos de Movimento'
    )

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.5
    )

    plt.tight_layout()

    path = os.path.join(
        PASTA_RESULTADOS,
        'movement_segment_durations.png'
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    return path


# ============================================================
# GRÁFICO: TIMELINE DA AUTONOMIA
# ============================================================

def plot_autonomy_timeline(
    times,
    speed,
    segments
):

    plt.figure(figsize=(13, 6))

    plt.plot(
        times,
        speed,
        linewidth=1.2,
        label='Velocidade agregada'
    )

    for i, segment in enumerate(segments):

        plt.axvspan(
            segment['start'],
            segment['end'],
            alpha=0.25,
            label='Movimento' if i == 0 else None
        )

    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade agregada (rad/s)')
    plt.title(
        'Janela de Autonomia e Segmentos de Movimento'
    )

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        PASTA_RESULTADOS,
        'autonomous_execution_timeline.png'
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    return path


# ============================================================
# SALVAR MÉTRICAS CINEMÁTICAS
# ============================================================

def save_kinematic_metrics(metrics):

    rows = []

    for j, name in enumerate(JUNTAS):

        rows.append([
            name,

            metrics['velocity_rms'][j],
            metrics['velocity_peak'][j],
            metrics['velocity_p95'][j],

            metrics['acceleration_rms'][j],
            metrics['acceleration_peak'][j],

            metrics['jerk_rms'][j],
            metrics['jerk_peak'][j],
            metrics['jerk_p95'][j]
        ])

    path = save_csv(

        'metricas_cinematicas_autonomia.csv',

        [
            'Joint',
            'Velocity_RMS_rad_s',
            'Velocity_Peak_rad_s',
            'Velocity_P95_rad_s',
            'Acceleration_RMS_rad_s2',
            'Acceleration_Peak_rad_s2',
            'Jerk_RMS_rad_s3',
            'Jerk_Peak_rad_s3',
            'Jerk_P95_rad_s3'
        ],

        rows
    )

    return path


# ============================================================
# SALVAR SEGMENTOS
# ============================================================

def save_segments(segments):

    rows = []

    for i, s in enumerate(segments):

        rows.append([
            i + 1,
            s['start'],
            s['end'],
            s['duration']
        ])

    return save_csv(

        'segmentos_movimento.csv',

        [
            'Segment',
            'Start_s',
            'End_s',
            'Duration_s'
        ],

        rows
    )


# ============================================================
# SALVAR TRANSIÇÕES
# ============================================================

def save_transitions(
    transition_results
):

    rows = []

    for r in transition_results:

        row = [
            r['transition'],
            r['time'],
            r['segment_before'],
            r['segment_after']
        ]

        row.extend(
            r['delta_v'].tolist()
        )

        rows.append(row)

    header = [
        'Transition',
        'Time_s',
        'Segment_Before',
        'Segment_After'
    ]

    header.extend([
        f'DeltaV_{j}'
        for j in JUNTAS
    ])

    return save_csv(
        'transicoes_segmentos_movimento.csv',
        header,
        rows
    )


# ============================================================
# SALVAR ESTATÍSTICAS DAS TRANSIÇÕES
# ============================================================

def save_transition_statistics(
    transition_results
):

    if len(transition_results) == 0:

        return None

    delta_v = np.array([
        r['delta_v']
        for r in transition_results
    ])

    rows = []

    for j, name in enumerate(JUNTAS):

        values = delta_v[:, j]

        rows.append([
            name,

            np.mean(values),
            rms(values),
            np.median(values),
            np.percentile(values, 95),
            np.max(values)
        ])

    return save_csv(

        'estatisticas_transicoes.csv',

        [
            'Joint',
            'Mean_Abs_DeltaV_rad_s',
            'RMS_DeltaV_rad_s',
            'Median_Abs_DeltaV_rad_s',
            'P95_Abs_DeltaV_rad_s',
            'Max_Abs_DeltaV_rad_s'
        ],

        rows
    )


# ============================================================
# NUVEM DE PONTOS
# ============================================================

def analyze_point_cloud(point_counts):

    print("\n" + "=" * 80)
    print("7. NUVEM DE PONTOS")
    print("=" * 80)

    if len(point_counts) == 0:

        print(
            "Nenhuma mensagem de nuvem de pontos encontrada."
        )

        return

    point_counts = np.asarray(
        point_counts
    )

    print(
        f"Pontos/frame - média : "
        f"{np.mean(point_counts):.0f}"
    )

    print(
        f"Pontos/frame - desvio: "
        f"{np.std(point_counts):.0f}"
    )

    print(
        f"Pontos/frame - mediana: "
        f"{np.median(point_counts):.0f}"
    )

    print(
        f"Pontos/frame - P5    : "
        f"{np.percentile(point_counts, 5):.0f}"
    )

    print(
        f"Pontos/frame - P95   : "
        f"{np.percentile(point_counts, 95):.0f}"
    )

    print(
        f"Pontos/frame - mínimo: "
        f"{np.min(point_counts):.0f}"
    )

    print(
        f"Pontos/frame - máximo: "
        f"{np.max(point_counts):.0f}"
    )

    save_csv(

        'estatisticas_nuvem_pontos.csv',

        [
            'Metric',
            'Value'
        ],

        [
            ['Mean', np.mean(point_counts)],
            ['Std', np.std(point_counts)],
            ['Median', np.median(point_counts)],
            ['P5', np.percentile(point_counts, 5)],
            ['P95', np.percentile(point_counts, 95)],
            ['Minimum', np.min(point_counts)],
            ['Maximum', np.max(point_counts)]
        ]
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("ANÁLISE CINEMÁTICA EXPERIMENTAL - ROS")
print("=" * 80)

print(f"\nArquivo: {ARQUIVO_BAG}")
print(f"Pasta de resultados: {PASTA_RESULTADOS}")

ensure_output_folder()

# ============================================================
# ABERTURA
# ============================================================

try:

    bag = rosbag.Bag(
        ARQUIVO_BAG
    )

except Exception as e:

    print(
        f"\nErro ao abrir rosbag: {e}"
    )

    exit()


# ============================================================
# LEITURA
# ============================================================

tempos_joint = []
velocidades = []

comandos = []

pontos_nuvem = []


print("\nLendo dados do rosbag...")


for topic, msg, t in bag.read_messages(
    topics=[
        TOPIC_JOINT_STATES,
        TOPIC_COMMAND,
        TOPIC_POINTS
    ]
):

    # --------------------------------------------------------
    # Joint states
    # --------------------------------------------------------

    if topic == TOPIC_JOINT_STATES:

        if len(msg.velocity) >= 6:

            timestamp = (
                msg.header.stamp.secs +
                msg.header.stamp.nsecs * 1e-9
            )

            tempos_joint.append(
                timestamp
            )

            velocidades.append(
                np.asarray(
                    msg.velocity[:6],
                    dtype=float
                )
            )

    # --------------------------------------------------------
    # Comandos
    # --------------------------------------------------------

    elif topic == TOPIC_COMMAND:

        duration = np.nan

        if len(msg.points) > 0:

            last_point = msg.points[-1]

            duration = (
                last_point.time_from_start.secs +
                last_point.time_from_start.nsecs * 1e-9
            )

        comandos.append({

            'receive': t.to_sec(),

            'duration': duration,

            'n_points': len(msg.points)
        })

    # --------------------------------------------------------
    # Nuvem de pontos
    # --------------------------------------------------------

    elif topic == TOPIC_POINTS:

        try:

            total_points = (
                msg.width *
                msg.height
            )

            if total_points > 0:

                pontos_nuvem.append(
                    total_points
                )

        except Exception:
            pass


bag.close()


# ============================================================
# CONVERSÃO
# ============================================================

tempos_joint = np.asarray(
    tempos_joint,
    dtype=float
)

velocidades = np.asarray(
    velocidades,
    dtype=float
)


# ============================================================
# LIMPEZA E ORDENAÇÃO
# ============================================================

if len(tempos_joint) < 20:

    print(
        "\nERRO: quantidade insuficiente de "
        "amostras de joint_states."
    )

    exit()


order = np.argsort(
    tempos_joint
)

tempos_joint = tempos_joint[order]
velocidades = velocidades[order]


dt_raw = np.diff(
    tempos_joint
)

valid = np.concatenate(
    (
        [True],
        dt_raw > 0
    )
)

tempos_joint = tempos_joint[valid]
velocidades = velocidades[valid]


# ============================================================
# AQUISIÇÃO
# ============================================================

duracao_aquisicao = (
    tempos_joint[-1] -
    tempos_joint[0]
)

dt_median = np.median(
    np.diff(tempos_joint)
)

taxa_amostragem = (
    1.0 / dt_median
)


print("\n" + "=" * 80)
print("1. AQUISIÇÃO CINEMÁTICA")
print("=" * 80)

print(
    f"Amostras de joint_states : "
    f"{len(tempos_joint)}"
)

print(
    f"Duração da aquisição     : "
    f"{duracao_aquisicao:.3f} s"
)

print(
    f"Taxa média de amostragem : "
    f"{taxa_amostragem:.2f} Hz"
)


# ============================================================
# DETECÇÃO DOS SEGMENTOS
# ============================================================

segments, speed, speed_smooth = (
    detect_movement_segments(
        tempos_joint,
        velocidades
    )
)


print("\n" + "=" * 80)
print("2. SEGMENTOS FÍSICOS DE MOVIMENTO")
print("=" * 80)


if len(segments) == 0:

    print(
        "Nenhum segmento de movimento foi detectado."
    )

    exit()


for i, s in enumerate(segments):

    print(
        f"Segmento {i+1:02d} | "
        f"{s['start']:.3f} - "
        f"{s['end']:.3f} s | "
        f"duração = "
        f"{s['duration']:.3f} s"
    )


# ============================================================
# JANELA DE AUTONOMIA
# ============================================================

autonomy_start = segments[0]['start']
autonomy_end = segments[-1]['end']

autonomy_duration = (
    autonomy_end -
    autonomy_start
)

movement_duration = sum(
    s['duration']
    for s in segments
)

movement_percentage = (
    100.0 *
    movement_duration /
    autonomy_duration
)


print("\n" + "=" * 80)
print("3. JANELA DE AUTONOMIA")
print("=" * 80)

print(
    f"Início da autonomia : "
    f"{autonomy_start:.3f} s"
)

print(
    f"Fim da autonomia    : "
    f"{autonomy_end:.3f} s"
)

print(
    f"Duração da janela   : "
    f"{autonomy_duration:.3f} s"
)

print(
    f"Tempo efetivo em movimento : "
    f"{movement_duration:.3f} s"
)

print(
    f"Percentual da janela em movimento : "
    f"{movement_percentage:.2f} %"
)


# ============================================================
# MÉTRICAS CINEMÁTICAS
# ============================================================

analysis = calculate_segment_metrics(
    tempos_joint,
    velocidades,
    segments
)


if analysis is None:

    print(
        "\nNão foi possível calcular métricas."
    )

    exit()


metrics = analysis['metrics']


print("\n" + "=" * 80)
print("4. MÉTRICAS CINEMÁTICAS DA AUTONOMIA")
print("=" * 80)


jerk_global = np.mean(
    metrics['jerk_rms']
)

velocity_global = np.mean(
    metrics['velocity_rms']
)

acceleration_global = np.mean(
    metrics['acceleration_rms']
)


print(
    f"\nJerk RMS médio das seis juntas: "
    f"{jerk_global:.4f} rad/s³"
)


for j, name in enumerate(JUNTAS):

    print(f"\n{name}")

    print(
        f"  Velocidade RMS : "
        f"{metrics['velocity_rms'][j]:.4f} rad/s"
    )

    print(
        f"  Velocidade pico: "
        f"{metrics['velocity_peak'][j]:.4f} rad/s"
    )

    print(
        f"  Velocidade P95 : "
        f"{metrics['velocity_p95'][j]:.4f} rad/s"
    )

    print(
        f"  Aceleração RMS : "
        f"{metrics['acceleration_rms'][j]:.4f} rad/s²"
    )

    print(
        f"  Aceleração pico: "
        f"{metrics['acceleration_peak'][j]:.4f} rad/s²"
    )

    print(
        f"  Jerk RMS       : "
        f"{metrics['jerk_rms'][j]:.4f} rad/s³"
    )

    print(
        f"  Jerk pico      : "
        f"{metrics['jerk_peak'][j]:.4f} rad/s³"
    )

    print(
        f"  Jerk P95       : "
        f"{metrics['jerk_p95'][j]:.4f} rad/s³"
    )


# ============================================================
# TRANSIÇÕES
# ============================================================

transition_results = analyze_transitions(
    tempos_joint,
    velocidades,
    segments
)


print("\n" + "=" * 80)
print("5. TRANSIÇÕES ENTRE SEGMENTOS FÍSICOS")
print("=" * 80)

print(
    f"Transições físicas analisadas: "
    f"{len(transition_results)}"
)


if len(transition_results) > 0:

    delta_v = np.array([
        r['delta_v']
        for r in transition_results
    ])

    for j, name in enumerate(JUNTAS):

        values = delta_v[:, j]

        print(
            f"\n{name}"
        )

        print(
            f"  |Δv| médio : "
            f"{np.mean(values):.5f} rad/s"
        )

        print(
            f"  Δv RMS     : "
            f"{rms(values):.5f} rad/s"
        )

        print(
            f"  Mediana    : "
            f"{np.median(values):.5f} rad/s"
        )

        print(
            f"  P95        : "
            f"{np.percentile(values, 95):.5f} rad/s"
        )

        print(
            f"  |Δv| máximo: "
            f"{np.max(values):.5f} rad/s"
        )


# ============================================================
# COMANDOS
# ============================================================

analyze_commands(
    comandos
)


# ============================================================
# NUVEM DE PONTOS
# ============================================================

analyze_point_cloud(
    pontos_nuvem
)


# ============================================================
# SALVAMENTO
# ============================================================

print("\n" + "=" * 80)
print("8. GERAÇÃO DOS RESULTADOS")
print("=" * 80)


path_metrics = save_kinematic_metrics(
    metrics
)

print(
    f"Arquivo gerado: {path_metrics}"
)


path_segments = save_segments(
    segments
)

print(
    f"Arquivo gerado: {path_segments}"
)


path_transitions = save_transitions(
    transition_results
)

print(
    f"Arquivo gerado: {path_transitions}"
)


path_transition_stats = (
    save_transition_statistics(
        transition_results
    )
)

if path_transition_stats:

    print(
        f"Arquivo gerado: "
        f"{path_transition_stats}"
    )


# ============================================================
# GRÁFICOS
# ============================================================

# Reconstroi eixo temporal uniforme somente para os
# segmentos físicos.

time_plot = []
velocity_plot = []

for segment in segments:

    t_u, v_u = resample_segment(
        tempos_joint,
        velocidades,
        segment['start'],
        segment['end']
    )

    if t_u is not None:

        v_f = safe_savgol(
            v_u,
            SG_WINDOW,
            SG_POLYORDER
        )

        time_plot.extend(
            t_u.tolist()
        )

        velocity_plot.extend(
            v_f.tolist()
        )


time_plot = np.asarray(
    time_plot
)

velocity_plot = np.asarray(
    velocity_plot
)


paths = []


paths.append(
    plot_velocity_profiles(
        time_plot,
        velocity_plot
    )
)

paths.append(
    plot_jerk_rms(
        metrics
    )
)

paths.append(
    plot_velocity_rms(
        metrics
    )
)

paths.append(
    plot_transition_analysis(
        transition_results
    )
)

paths.append(
    plot_segment_durations(
        segments
    )
)

paths.append(
    plot_autonomy_timeline(
        tempos_joint,
        speed_smooth,
        segments
    )
)


for p in paths:

    if p is not None:

        print(
            f"Gráfico gerado: {p}"
        )


# ============================================================
# RESUMO EXPERIMENTAL
# ============================================================

summary_path = os.path.join(
    PASTA_RESULTADOS,
    'resumo_experimental.txt'
)


with open(
    summary_path,
    'w'
) as f:

    f.write(
        "RESUMO EXPERIMENTAL - AUTONOMIA COMPARTILHADA\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Arquivo: {ARQUIVO_BAG}\n"
    )

    f.write(
        f"Amostras joint_states: "
        f"{len(tempos_joint)}\n"
    )

    f.write(
        f"Taxa de aquisição: "
        f"{taxa_amostragem:.2f} Hz\n"
    )

    f.write(
        f"Duração total da aquisição: "
        f"{duracao_aquisicao:.3f} s\n"
    )

    f.write(
        f"Início da autonomia: "
        f"{autonomy_start:.3f} s\n"
    )

    f.write(
        f"Fim da autonomia: "
        f"{autonomy_end:.3f} s\n"
    )

    f.write(
        f"Duração da janela de autonomia: "
        f"{autonomy_duration:.3f} s\n"
    )

    f.write(
        f"Tempo efetivo em movimento: "
        f"{movement_duration:.3f} s\n"
    )

    f.write(
        f"Percentual em movimento: "
        f"{movement_percentage:.2f} %\n"
    )

    f.write(
        f"Segmentos físicos: "
        f"{len(segments)}\n"
    )

    f.write(
        f"Transições físicas: "
        f"{len(transition_results)}\n"
    )

    f.write(
        f"Velocidade RMS média: "
        f"{velocity_global:.6f} rad/s\n"
    )

    f.write(
        f"Aceleração RMS média: "
        f"{acceleration_global:.6f} rad/s2\n"
    )

    f.write(
        f"Jerk RMS médio: "
        f"{jerk_global:.6f} rad/s3\n"
    )

    f.write("\n")

    f.write(
        "METODOLOGIA:\n"
    )

    f.write(
        "As métricas cinemáticas foram calculadas exclusivamente "
        "nos segmentos físicos de movimento detectados a partir "
        "dos dados de /joint_states. Os sinais foram reamostrados "
        "em uma grade temporal uniforme de 50 Hz e suavizados "
        "com filtro Savitzky-Golay antes das diferenciações "
        "numéricas. O jerk foi obtido pela segunda diferenciação "
        "da velocidade articular.\n\n"
    )

    f.write(
        "A análise de transição considera a diferença entre "
        "as velocidades médias observadas em pequenas janelas "
        "temporais imediatamente antes e depois de cada "
        "transição entre segmentos físicos.\n"
    )


print(
    f"Arquivo gerado: {summary_path}"
)


# ============================================================
# RESUMO NO TERMINAL
# ============================================================

print("\n" + "=" * 80)
print("RESUMO EXPERIMENTAL")
print("=" * 80)

print(
    f"\nJanela de autonomia       : "
    f"{autonomy_duration:.3f} s"
)

print(
    f"Tempo efetivo em movimento: "
    f"{movement_duration:.3f} s"
)

print(
    f"Segmentos físicos        : "
    f"{len(segments)}"
)

print(
    f"Transições físicas       : "
    f"{len(transition_results)}"
)

print(
    f"Velocidade RMS média     : "
    f"{velocity_global:.4f} rad/s"
)

print(
    f"Aceleração RMS média     : "
    f"{acceleration_global:.4f} rad/s²"
)

print(
    f"Jerk RMS médio            : "
    f"{jerk_global:.4f} rad/s³"
)

print(
    f"Mensagens /command        : "
    f"{len(comandos)}"
)


print("\n" + "=" * 80)

print(
    "OBSERVAÇÕES METODOLÓGICAS:"
)

print(
    "1. As métricas cinemáticas são calculadas apenas "
    "durante os segmentos físicos de movimento."
)

print(
    "2. Os períodos de repouso não são utilizados para "
    "reduzir artificialmente o jerk RMS."
)

print(
    "3. A quantidade de mensagens /command não é "
    "interpretada como quantidade de trajetórias físicas."
)

print(
    "4. As transições são determinadas a partir do "
    "comportamento observado em /joint_states."
)

print(
    "5. A variação de velocidade nas transições é "
    "calculada utilizando médias temporais antes e "
    "depois da transição."
)

print(
    "6. A janela de autonomia é definida pelo primeiro "
    "e último segmento físico detectado."
)

print(
    "7. Eventos de alto nível como Alignment, Descent, "
    "Grasp e Lifting não são inferidos a partir do ROS "
    "quando seus timestamps não estão disponíveis."
)

print(
    "\nANÁLISE CONCLUÍDA."
)

print("=" * 80)





















# #!/usr/bin/env python3
# # script atualizado_1
# import rosbag
# import numpy as np
# import matplotlib.pyplot as plt

# from scipy.signal import savgol_filter


# # ============================================================
# # CONFIGURAÇÕES
# # ============================================================

# # arquivo_bag = 'ENSAIO GRASP/ensaio_grasping_0.bag'
# arquivo_bag = 'ensaio_grasping_0.bag'

# TOPIC_JOINT_STATES = '/ur5/joint_states'
# TOPIC_COMMAND = '/ur5/eff_joint_traj_controller/command'

# JUNTAS = [
#     'Shoulder Pan',
#     'Shoulder Lift',
#     'Elbow',
#     'Wrist 1',
#     'Wrist 2',
#     'Wrist 3'
# ]

# # Parâmetros da filtragem
# SG_WINDOW = 21
# SG_POLYORDER = 3

# # Janela utilizada para comparar velocidades
# # antes/depois de cada transição
# TRANSITION_WINDOW = 0.20  # segundos


# # ============================================================
# # FUNÇÕES AUXILIARES
# # ============================================================

# def timestamp_to_sec(stamp):
#     """
#     Converte um rospy.Time para segundos.
#     """
#     return stamp.secs + stamp.nsecs * 1e-9


# def safe_header_stamp(msg):
#     """
#     Retorna o timestamp do header caso seja válido.
#     """
#     try:
#         stamp = timestamp_to_sec(msg.header.stamp)

#         if stamp > 0:
#             return stamp

#     except Exception:
#         pass

#     return None


# def rms(x):
#     """
#     Root Mean Square.
#     """
#     x = np.asarray(x)

#     if len(x) == 0:
#         return np.nan

#     return np.sqrt(np.mean(x ** 2))


# # ============================================================
# # ABERTURA DO ROSBAG
# # ============================================================

# print("=" * 70)
# print("ANÁLISE CINEMÁTICA EXPERIMENTAL - ROS")
# print("=" * 70)

# print(f"\nArquivo: {arquivo_bag}\n")

# try:
#     bag = rosbag.Bag(arquivo_bag)

# except Exception as e:
#     print(f"Erro ao abrir o rosbag: {e}")
#     exit()


# # ============================================================
# # LEITURA DOS DADOS
# # ============================================================

# tempos_joint = []
# velocidades = []

# trajetorias = []


# for topic, msg, t in bag.read_messages(
#         topics=[TOPIC_JOINT_STATES, TOPIC_COMMAND]):

#     # --------------------------------------------------------
#     # JOINT STATES
#     # --------------------------------------------------------

#     if topic == TOPIC_JOINT_STATES:

#         if len(msg.velocity) >= 6:

#             timestamp = timestamp_to_sec(msg.header.stamp)

#             tempos_joint.append(timestamp)

#             velocidades.append(
#                 np.array(msg.velocity[:6], dtype=float)
#             )


#     # --------------------------------------------------------
#     # TRAJETÓRIAS AUTÔNOMAS
#     # --------------------------------------------------------

#     elif topic == TOPIC_COMMAND:

#         receive_time = t.to_sec()

#         header_time = safe_header_stamp(msg)

#         if header_time is not None:
#             trajectory_start = header_time
#             timestamp_source = 'header'
#         else:
#             trajectory_start = receive_time
#             timestamp_source = 'bag'

#         duration = np.nan

#         if len(msg.points) > 0:

#             last_point = msg.points[-1]

#             duration = (
#                 last_point.time_from_start.secs
#                 + last_point.time_from_start.nsecs * 1e-9
#             )

#         trajectory_end = (
#             trajectory_start + duration
#             if np.isfinite(duration)
#             else np.nan
#         )

#         trajetorias.append({
#             'start': trajectory_start,
#             'end': trajectory_end,
#             'duration': duration,
#             'receive': receive_time,
#             'source': timestamp_source,
#             'n_points': len(msg.points)
#         })


# bag.close()


# # ============================================================
# # CONVERSÃO PARA NUMPY
# # ============================================================

# tempos_joint = np.asarray(tempos_joint)
# velocidades = np.asarray(velocidades)


# # ============================================================
# # VERIFICAÇÃO DOS DADOS
# # ============================================================

# print("Amostras de joint_states :", len(tempos_joint))

# if len(tempos_joint) < 10:

#     print("\nERRO: quantidade insuficiente de joint_states.")
#     exit()


# # Ordenação temporal

# ordem = np.argsort(tempos_joint)

# tempos_joint = tempos_joint[ordem]
# velocidades = velocidades[ordem]


# # Remove timestamps duplicados

# dt_raw = np.diff(tempos_joint)

# mask_unique = np.concatenate(
#     ([True], dt_raw > 0)
# )

# tempos_joint = tempos_joint[mask_unique]
# velocidades = velocidades[mask_unique]


# duracao_aquisicao = tempos_joint[-1] - tempos_joint[0]

# taxa_amostragem = (
#     1.0 / np.median(np.diff(tempos_joint))
# )


# print(f"Duração da aquisição     : {duracao_aquisicao:.3f} s")
# print(f"Taxa média de amostragem : {taxa_amostragem:.2f} Hz")


# # ============================================================
# # REAMOSTRAGEM TEMPORAL
# # ============================================================

# dt_uniforme = 1.0 / 50.0

# tempo_uniforme = np.arange(
#     tempos_joint[0],
#     tempos_joint[-1],
#     dt_uniforme
# )

# velocidades_uniformes = np.zeros(
#     (len(tempo_uniforme), 6)
# )


# for j in range(6):

#     velocidades_uniformes[:, j] = np.interp(
#         tempo_uniforme,
#         tempos_joint,
#         velocidades[:, j]
#     )


# # ============================================================
# # FILTRO SAVITZKY-GOLAY
# # ============================================================

# if len(tempo_uniforme) > SG_WINDOW:

#     velocidades_filtradas = savgol_filter(
#         velocidades_uniformes,
#         SG_WINDOW,
#         SG_POLYORDER,
#         axis=0
#     )

# else:

#     velocidades_filtradas = velocidades_uniformes.copy()


# print(
#     f"Filtro Savitzky-Golay aplicado: "
#     f"janela={SG_WINDOW}, ordem={SG_POLYORDER}"
# )


# # ============================================================
# # ACELERAÇÃO E JERK
# # ============================================================

# aceleracoes = np.gradient(
#     velocidades_filtradas,
#     dt_uniforme,
#     axis=0
# )

# jerk = np.gradient(
#     aceleracoes,
#     dt_uniforme,
#     axis=0
# )


# # ============================================================
# # JERK RMS
# # ============================================================

# jerk_rms = np.sqrt(
#     np.mean(jerk ** 2, axis=0)
# )

# jerk_global = np.mean(jerk_rms)


# print("\n" + "-" * 70)
# print("ANÁLISE DE SUAVIDADE CINEMÁTICA")
# print("-" * 70)

# print(
#     f"\nJerk RMS global "
#     f"(média das seis juntas): "
#     f"{jerk_global:.4f} rad/s³"
# )

# for nome, valor in zip(JUNTAS, jerk_rms):

#     jerk_pico = np.max(np.abs(jerk[:, JUNTAS.index(nome)]))

#     print(
#         f"{nome:18s} | "
#         f"Jerk RMS = {valor:10.4f} rad/s³ | "
#         f"Jerk pico = {jerk_pico:10.4f} rad/s³"
#     )


# # ============================================================
# # VELOCIDADE RMS E PICO
# # ============================================================

# print("\n" + "-" * 70)
# print("VELOCIDADE ARTICULAR")
# print("-" * 70)

# for j, nome in enumerate(JUNTAS):

#     v_rms = rms(velocidades_filtradas[:, j])
#     v_peak = np.max(np.abs(velocidades_filtradas[:, j]))

#     print(
#         f"{nome:18s} | "
#         f"RMS = {v_rms:8.4f} rad/s | "
#         f"Pico = {v_peak:8.4f} rad/s"
#     )


# # ============================================================
# # ACELERAÇÃO RMS
# # ============================================================

# print("\n" + "-" * 70)
# print("ACELERAÇÃO ARTICULAR")
# print("-" * 70)

# for j, nome in enumerate(JUNTAS):

#     a_rms = rms(aceleracoes[:, j])

#     print(
#         f"{nome:18s} | "
#         f"RMS = {a_rms:8.4f} rad/s²"
#     )


# # ============================================================
# # ANÁLISE DAS TRAJETÓRIAS
# # ============================================================

# print("\n" + "=" * 70)
# print("ANÁLISE DAS TRAJETÓRIAS AUTÔNOMAS")
# print("=" * 70)

# print(
#     f"\nMensagens /command recebidas: "
#     f"{len(trajetorias)}"
# )


# if len(trajetorias) == 0:

#     print("\nNenhuma trajetória encontrada.")

# else:

#     for i, traj in enumerate(trajetorias):

#         print(
#             f"\nTrajetória/comando {i+1:02d}: "
#             f"{traj['n_points']} pontos | "
#             f"duração = {traj['duration']:.3f} s | "
#             f"origem temporal = {traj['source']}"
#         )


# # ============================================================
# # JANELA PLANEJADA DAS TRAJETÓRIAS
# # ============================================================

# duracoes_validas = [
#     t['duration']
#     for t in trajetorias
#     if np.isfinite(t['duration'])
# ]


# if len(duracoes_validas) > 0:

#     soma_duracoes = np.sum(duracoes_validas)

#     print(
#         f"\nSoma das durações planejadas: "
#         f"{soma_duracoes:.3f} s"
#     )


# # ============================================================
# # JANELA TEMPORAL ENTRE PRIMEIRA E ÚLTIMA TRAJETÓRIA
# # ============================================================

# starts = [
#     t['start']
#     for t in trajetorias
#     if np.isfinite(t['start'])
# ]

# ends = [
#     t['end']
#     for t in trajetorias
#     if np.isfinite(t['end'])
# ]


# if len(starts) > 0 and len(ends) > 0:

#     inicio_planejado = min(starts)
#     fim_planejado = max(ends)

#     janela_total = (
#         fim_planejado - inicio_planejado
#     )

#     print(
#         f"Janela temporal total planejada: "
#         f"{janela_total:.3f} s"
#     )


# # ============================================================
# # ANÁLISE DAS TRANSIÇÕES ENTRE TRAJETÓRIAS
# # ============================================================

# print("\n" + "=" * 70)
# print("ANÁLISE DAS TRANSIÇÕES ENTRE TRAJETÓRIAS")
# print("=" * 70)


# transicoes = []


# if len(trajetorias) >= 2:

#     # Ordena cronologicamente

#     trajetorias = sorted(
#         trajetorias,
#         key=lambda x: x['start']
#     )


#     for i in range(len(trajetorias) - 1):

#         atual = trajetorias[i]
#         proxima = trajetorias[i + 1]

#         if not np.isfinite(atual['end']):
#             continue

#         t_transicao = atual['end']

#         t_proxima = proxima['start']

#         gap = t_proxima - t_transicao


#         # ----------------------------------------------------
#         # Seleciona janela antes da transição
#         # ----------------------------------------------------

#         mask_before = (
#             (tempo_uniforme >= t_transicao - TRANSITION_WINDOW) &
#             (tempo_uniforme < t_transicao)
#         )


#         # ----------------------------------------------------
#         # Seleciona janela depois da transição
#         # ----------------------------------------------------

#         mask_after = (
#             (tempo_uniforme >= t_transicao) &
#             (tempo_uniforme <=
#              t_transicao + TRANSITION_WINDOW)
#         )


#         if (
#             np.sum(mask_before) < 3 or
#             np.sum(mask_after) < 3
#         ):

#             print(
#                 f"\nTransição {i+1:02d} -> {i+2:02d}: "
#                 f"dados insuficientes"
#             )

#             continue


#         v_before = np.mean(
#             velocidades_filtradas[mask_before],
#             axis=0
#         )

#         v_after = np.mean(
#             velocidades_filtradas[mask_after],
#             axis=0
#         )


#         delta_v = v_after - v_before

#         delta_norm = np.linalg.norm(delta_v)

#         delta_max = np.max(np.abs(delta_v))

#         delta_rms = rms(delta_v)


#         transicoes.append({

#             'index': i + 1,

#             'time': t_transicao,

#             'gap': gap,

#             'delta_v': delta_v,

#             'delta_norm': delta_norm,

#             'delta_max': delta_max,

#             'delta_rms': delta_rms
#         })


#         print(
#             f"\nTransição {i+1:02d} -> {i+2:02d}"
#         )

#         print(
#             f"  Instante planejado : "
#             f"{t_transicao:.3f} s"
#         )

#         print(
#             f"  Gap entre comandos : "
#             f"{gap:.4f} s"
#         )

#         print(
#             f"  ||Δv||             : "
#             f"{delta_norm:.4f} rad/s"
#         )

#         print(
#             f"  RMS(Δv)            : "
#             f"{delta_rms:.4f} rad/s"
#         )

#         print(
#             f"  max(|Δv|)          : "
#             f"{delta_max:.4f} rad/s"
#         )

#         print("  Δv por junta:")

#         for nome, dv in zip(JUNTAS, delta_v):

#             print(
#                 f"     {nome:18s}: "
#                 f"{dv:+.4f} rad/s"
#             )


# # ============================================================
# # ESTATÍSTICAS DAS TRANSIÇÕES
# # ============================================================

# if len(transicoes) > 0:

#     normas = np.array([
#         t['delta_norm']
#         for t in transicoes
#     ])

#     maximos = np.array([
#         t['delta_max']
#         for t in transicoes
#     ])

#     rms_transicoes = np.array([
#         t['delta_rms']
#         for t in transicoes
#     ])


#     print("\n" + "-" * 70)
#     print("RESUMO DAS TRANSIÇÕES")
#     print("-" * 70)

#     print(
#         f"Número de transições analisadas : "
#         f"{len(transicoes)}"
#     )

#     print(
#         f"||Δv|| médio                    : "
#         f"{np.mean(normas):.4f} rad/s"
#     )

#     print(
#         f"||Δv|| máximo                   : "
#         f"{np.max(normas):.4f} rad/s"
#     )

#     print(
#         f"max(|Δv|) médio                 : "
#         f"{np.mean(maximos):.4f} rad/s"
#     )

#     print(
#         f"max(|Δv|) máximo                : "
#         f"{np.max(maximos):.4f} rad/s"
#     )

#     print(
#         f"RMS médio das transições        : "
#         f"{np.mean(rms_transicoes):.4f} rad/s"
#     )


# # ============================================================
# # GRÁFICO DE VELOCIDADE
# # ============================================================

# plt.figure(figsize=(12, 6))

# for j, nome in enumerate(JUNTAS):

#     plt.plot(
#         tempo_uniforme - tempo_uniforme[0],
#         velocidades_filtradas[:, j],
#         label=nome,
#         linewidth=1.5
#     )


# plt.title(
#     'Joint Velocity Profiles During Autonomous Motion'
# )

# plt.xlabel('Time (s)')

# plt.ylabel('Joint velocity (rad/s)')

# plt.legend()

# plt.grid(
#     True,
#     linestyle='--',
#     alpha=0.7
# )

# plt.tight_layout()

# plt.savefig(
#     'joint_velocity_profiles.png',
#     dpi=300
# )

# plt.close()


# # ============================================================
# # GRÁFICO DAS TRANSIÇÕES
# # ============================================================

# if len(transicoes) > 0:

#     tempos_transicao = [
#         t['time'] - tempo_uniforme[0]
#         for t in transicoes
#     ]

#     delta_norms = [
#         t['delta_norm']
#         for t in transicoes
#     ]


#     plt.figure(figsize=(10, 5))

#     plt.plot(
#         tempos_transicao,
#         delta_norms,
#         marker='o',
#         linewidth=1.5
#     )

#     plt.xlabel('Transition time (s)')

#     plt.ylabel(r'$\|\Delta v\|$ (rad/s)')

#     plt.title(
#         'Velocity Discontinuity at Trajectory Transitions'
#     )

#     plt.grid(
#         True,
#         linestyle='--',
#         alpha=0.7
#     )

#     plt.tight_layout()

#     plt.savefig(
#         'trajectory_transition_analysis.png',
#         dpi=300
#     )

#     plt.close()


# # ============================================================
# # FINAL
# # ============================================================

# print("\n" + "=" * 70)

# print(
#     "OBSERVAÇÃO METODOLÓGICA:"
# )

# print(
#     "O Jerk RMS foi calculado a partir da velocidade "
#     "articular reamostrada em uma grade temporal uniforme "
#     "e suavizada antes das diferenciações numéricas."
# )

# print(
#     "As mensagens /command são tratadas como segmentos "
#     "de trajetória planejados. A quantidade de mensagens "
#     "não é interpretada isoladamente como quantidade de "
#     "movimentos físicos."
# )

# print(
#     "A análise de transição compara a velocidade articular "
#     "média imediatamente antes e depois do instante final "
#     "planejado de cada segmento."
# )

# print(
#     "Os resultados de transição devem ser interpretados "
#     "como indicadores de continuidade cinemática e não "
#     "como medida direta de jerk."
# )

# print("\nAnálise concluída.")
# print("=" * 70)

















# #!/usr/bin/env python3

# import rosbag
# import numpy as np
# import matplotlib.pyplot as plt

# try:
#     from scipy.signal import savgol_filter
#     SCIPY_AVAILABLE = True
# except ImportError:
#     SCIPY_AVAILABLE = False


# # ============================================================
# # CONFIGURAÇÃO
# # ============================================================

# ARQUIVO_BAG = 'ENSAIO GRASP/ensaio_grasping_0.bag'

# TOPIC_JOINT_STATES = '/ur5/joint_states'
# TOPIC_TRAJECTORY_COMMAND = '/ur5/eff_joint_traj_controller/command'

# JOINT_NAMES = [
#     'Shoulder Pan',
#     'Shoulder Lift',
#     'Elbow',
#     'Wrist 1',
#     'Wrist 2',
#     'Wrist 3'
# ]

# # Frequência utilizada para reamostragem da velocidade.
# # Ajuste conforme a taxa real de publicação de /joint_states.
# RESAMPLE_FREQUENCY = 100.0  # Hz

# # Parâmetros do filtro Savitzky-Golay.
# # A janela precisa ser ímpar.
# SAVGOL_WINDOW = 21
# SAVGOL_POLYORDER = 3

# # Remove intervalos de amostragem excessivamente pequenos.
# MIN_DT = 0.001  # s

# # Limite para considerar que o robô está efetivamente em movimento.
# VELOCITY_THRESHOLD = 0.005  # rad/s


# # ============================================================
# # FUNÇÕES AUXILIARES
# # ============================================================

# def safe_float(value):
#     try:
#         return float(value)
#     except Exception:
#         return np.nan


# def trajectory_time_to_seconds(duration):
#     """
#     Converte rospy.Duration para segundos.
#     """
#     try:
#         return duration.to_sec()
#     except Exception:
#         return 0.0


# def rms(signal):
#     signal = np.asarray(signal)

#     if len(signal) == 0:
#         return np.nan

#     return np.sqrt(np.mean(signal ** 2))


# def percentile(signal, p):
#     if len(signal) == 0:
#         return np.nan

#     return np.percentile(signal, p)


# # ============================================================
# # LEITURA DO ROSBAG
# # ============================================================

# print("=" * 72)
# print("ANÁLISE CINEMÁTICA EXPERIMENTAL - ROS")
# print("=" * 72)
# print(f"Arquivo: {ARQUIVO_BAG}")
# print()

# tempos_joint = []
# velocidades_joint = []

# trajectory_commands = []


# try:

#     bag = rosbag.Bag(ARQUIVO_BAG)

# except Exception as e:

#     print(f"ERRO ao abrir o rosbag: {e}")
#     raise SystemExit


# for topic, msg, t in bag.read_messages(
#         topics=[
#             TOPIC_JOINT_STATES,
#             TOPIC_TRAJECTORY_COMMAND
#         ]):

#     # ========================================================
#     # JOINT STATES
#     # ========================================================

#     if topic == TOPIC_JOINT_STATES:

#         if len(msg.velocity) >= 6:

#             timestamp = msg.header.stamp.to_sec()

#             velocities = np.asarray(
#                 msg.velocity[:6],
#                 dtype=float
#             )

#             if np.all(np.isfinite(velocities)):

#                 tempos_joint.append(timestamp)
#                 velocidades_joint.append(velocities)

#     # ========================================================
#     # TRAJECTORY COMMAND
#     # ========================================================

#     elif topic == TOPIC_TRAJECTORY_COMMAND:

#         command_time = t.to_sec()

#         # Número de pontos da trajetória
#         n_points = len(msg.points)

#         # Duração planejada da trajetória
#         planned_duration = 0.0

#         if n_points > 0:

#             planned_duration = max(
#                 trajectory_time_to_seconds(
#                     point.time_from_start
#                 )
#                 for point in msg.points
#             )

#         # Timestamp do início planejado
#         header_time = 0.0

#         try:
#             header_time = msg.header.stamp.to_sec()
#         except Exception:
#             header_time = 0.0

#         if header_time <= 0:
#             header_time = command_time

#         trajectory_commands.append({
#             'bag_time': command_time,
#             'start_time': header_time,
#             'duration': planned_duration,
#             'end_time': header_time + planned_duration,
#             'points': n_points
#         })


# bag.close()


# # ============================================================
# # VALIDAÇÃO DOS DADOS
# # ============================================================

# if len(tempos_joint) < 10:

#     print("ERRO: quantidade insuficiente de amostras de joint_states.")
#     raise SystemExit


# tempos_joint = np.asarray(tempos_joint)
# velocidades_joint = np.asarray(velocidades_joint)


# # ============================================================
# # ORDENAÇÃO TEMPORAL
# # ============================================================

# ordem = np.argsort(tempos_joint)

# tempos_joint = tempos_joint[ordem]
# velocidades_joint = velocidades_joint[ordem]


# # ============================================================
# # REMOÇÃO DE TIMESTAMPS DUPLICADOS
# # ============================================================

# tempos_unicos, indices_unicos = np.unique(
#     tempos_joint,
#     return_index=True
# )

# velocidades_unicas = velocidades_joint[indices_unicos]

# tempos_joint = tempos_unicos
# velocidades_joint = velocidades_unicas


# # ============================================================
# # NORMALIZAÇÃO DO TEMPO
# # ============================================================

# tempos_joint = tempos_joint - tempos_joint[0]


# # ============================================================
# # ESTATÍSTICAS DA AQUISIÇÃO
# # ============================================================

# dt_original = np.diff(tempos_joint)

# dt_validos = dt_original[
#     dt_original >= MIN_DT
# ]

# if len(dt_validos) == 0:

#     print("ERRO: não existem intervalos temporais válidos.")
#     raise SystemExit


# taxa_amostragem_media = 1.0 / np.mean(dt_validos)

# print("=" * 72)
# print("AQUISIÇÃO DE DADOS")
# print("=" * 72)

# print(
#     f"Amostras de joint_states : "
#     f"{len(tempos_joint)}"
# )

# print(
#     f"Duração da aquisição     : "
#     f"{tempos_joint[-1]:.3f} s"
# )

# print(
#     f"Taxa média de amostragem : "
#     f"{taxa_amostragem_media:.2f} Hz"
# )


# # ============================================================
# # REAMOSTRAGEM EM GRADE TEMPORAL UNIFORME
# # ============================================================

# dt_uniforme = 1.0 / RESAMPLE_FREQUENCY

# tempo_uniforme = np.arange(
#     tempos_joint[0],
#     tempos_joint[-1],
#     dt_uniforme
# )

# velocidades_uniformes = np.zeros(
#     (len(tempo_uniforme), 6)
# )


# for j in range(6):

#     velocidades_uniformes[:, j] = np.interp(
#         tempo_uniforme,
#         tempos_joint,
#         velocidades_joint[:, j]
#     )


# # ============================================================
# # FILTRAGEM PARA ANÁLISE DIFERENCIAL
# # ============================================================

# velocidades_filtradas = np.copy(
#     velocidades_uniformes
# )

# if SCIPY_AVAILABLE:

#     # Garante que a janela seja válida
#     window = SAVGOL_WINDOW

#     if window >= len(tempo_uniforme):
#         window = len(tempo_uniforme) - 1

#     if window % 2 == 0:
#         window -= 1

#     if window > SAVGOL_POLYORDER:

#         for j in range(6):

#             velocidades_filtradas[:, j] = savgol_filter(
#                 velocidades_uniformes[:, j],
#                 window_length=window,
#                 polyorder=SAVGOL_POLYORDER
#             )

#         print(
#             f"Filtro Savitzky-Golay aplicado: "
#             f"janela={window}, "
#             f"ordem={SAVGOL_POLYORDER}"
#         )

# else:

#     print(
#         "AVISO: scipy não disponível. "
#         "A análise será realizada sem filtragem."
#     )


# # ============================================================
# # ACELERAÇÃO
# # ============================================================

# aceleracoes = np.zeros_like(
#     velocidades_filtradas
# )

# for j in range(6):

#     aceleracoes[:, j] = np.gradient(
#         velocidades_filtradas[:, j],
#         tempo_uniforme
#     )


# # ============================================================
# # JERK
# # ============================================================

# jerk = np.zeros_like(
#     aceleracoes
# )

# for j in range(6):

#     jerk[:, j] = np.gradient(
#         aceleracoes[:, j],
#         tempo_uniforme
#     )


# # ============================================================
# # MÉTRICAS CINEMÁTICAS
# # ============================================================

# jerk_rms = np.zeros(6)
# jerk_peak = np.zeros(6)
# velocity_rms = np.zeros(6)
# velocity_peak = np.zeros(6)
# acceleration_rms = np.zeros(6)

# for j in range(6):

#     jerk_rms[j] = rms(jerk[:, j])

#     jerk_peak[j] = np.max(
#         np.abs(jerk[:, j])
#     )

#     velocity_rms[j] = rms(
#         velocidades_filtradas[:, j]
#     )

#     velocity_peak[j] = np.max(
#         np.abs(velocidades_filtradas[:, j])
#     )

#     acceleration_rms[j] = rms(
#         aceleracoes[:, j]
#     )


# jerk_global_rms = np.mean(jerk_rms)


# # ============================================================
# # RESULTADOS DE JERK
# # ============================================================

# print()
# print("=" * 72)
# print("ANÁLISE DE SUAVIDADE CINEMÁTICA")
# print("=" * 72)

# print(
#     f"Jerk RMS global "
#     f"(média das seis juntas): "
#     f"{jerk_global_rms:.4f} rad/s³"
# )

# print()

# for i, nome in enumerate(JOINT_NAMES):

#     print(
#         f"{nome:18s} | "
#         f"Jerk RMS = {jerk_rms[i]:10.4f} rad/s³ | "
#         f"Jerk pico = {jerk_peak[i]:10.4f} rad/s³"
#     )


# # ============================================================
# # VELOCIDADES
# # ============================================================

# print()
# print("=" * 72)
# print("VELOCIDADE ARTICULAR")
# print("=" * 72)

# for i, nome in enumerate(JOINT_NAMES):

#     print(
#         f"{nome:18s} | "
#         f"RMS = {velocity_rms[i]:8.4f} rad/s | "
#         f"Pico = {velocity_peak[i]:8.4f} rad/s"
#     )


# # ============================================================
# # ACELERAÇÃO
# # ============================================================

# print()
# print("=" * 72)
# print("ACELERAÇÃO ARTICULAR")
# print("=" * 72)

# for i, nome in enumerate(JOINT_NAMES):

#     print(
#         f"{nome:18s} | "
#         f"RMS = {acceleration_rms[i]:8.4f} rad/s²"
#     )


# # ============================================================
# # ANÁLISE DOS COMANDOS DE TRAJETÓRIA
# # ============================================================

# print()
# print("=" * 72)
# print("COMANDOS DE TRAJETÓRIA")
# print("=" * 72)

# print(
#     f"Mensagens /command recebidas: "
#     f"{len(trajectory_commands)}"
# )

# if len(trajectory_commands) > 0:

#     inicio_planejado = min(
#         x['start_time']
#         for x in trajectory_commands
#     )

#     fim_planejado = max(
#         x['end_time']
#         for x in trajectory_commands
#     )

#     duracao_planejada = (
#         fim_planejado - inicio_planejado
#     )

#     print(
#         f"Janela planejada pelas trajetórias: "
#         f"{duracao_planejada:.3f} s"
#     )

#     print()

#     for i, command in enumerate(
#             trajectory_commands, start=1):

#         print(
#             f"Trajetória/comando {i:02d}: "
#             f"{command['points']} pontos | "
#             f"duração = "
#             f"{command['duration']:.3f} s"
#         )


# # ============================================================
# # IDENTIFICAÇÃO DE MOVIMENTO
# # ============================================================

# velocidade_norma = np.linalg.norm(
#     velocidades_filtradas,
#     axis=1
# )

# indices_movimento = (
#     velocidade_norma > VELOCITY_THRESHOLD
# )

# if np.any(indices_movimento):

#     tempo_inicio_movimento = (
#         tempo_uniforme[
#             np.where(indices_movimento)[0][0]
#         ]
#     )

#     tempo_fim_movimento = (
#         tempo_uniforme[
#             np.where(indices_movimento)[0][-1]
#         ]
#     )

#     duracao_movimento = (
#         tempo_fim_movimento -
#         tempo_inicio_movimento
#     )

#     print()
#     print("=" * 72)
#     print("MOVIMENTO OBSERVADO")
#     print("=" * 72)

#     print(
#         f"Início aproximado do movimento: "
#         f"{tempo_inicio_movimento:.3f} s"
#     )

#     print(
#         f"Fim aproximado do movimento: "
#         f"{tempo_fim_movimento:.3f} s"
#     )

#     print(
#         f"Duração aproximada do movimento: "
#         f"{duracao_movimento:.3f} s"
#     )


# # ============================================================
# # GRÁFICO DE VELOCIDADES
# # ============================================================

# plt.figure(figsize=(11, 6))

# plt.plot(
#     tempo_uniforme,
#     velocidades_filtradas[:, 0],
#     linewidth=1.8,
#     label='Shoulder Pan'
# )

# plt.plot(
#     tempo_uniforme,
#     velocidades_filtradas[:, 1],
#     linewidth=1.8,
#     label='Shoulder Lift'
# )

# plt.plot(
#     tempo_uniforme,
#     velocidades_filtradas[:, 2],
#     linewidth=1.8,
#     label='Elbow'
# )

# plt.plot(
#     tempo_uniforme,
#     velocidades_filtradas[:, 3],
#     linewidth=1.8,
#     label='Wrist 1'
# )

# plt.plot(
#     tempo_uniforme,
#     velocidades_filtradas[:, 4],
#     linewidth=1.8,
#     label='Wrist 2'
# )

# plt.plot(
#     tempo_uniforme,
#     velocidades_filtradas[:, 5],
#     linewidth=1.8,
#     label='Wrist 3'
# )

# plt.xlabel('Time (s)')
# plt.ylabel('Joint velocity (rad/s)')

# plt.title(
#     'Joint Velocity Profiles During Autonomous Motion'
# )

# plt.grid(
#     True,
#     linestyle='--',
#     alpha=0.5
# )

# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     'joint_velocity_profiles.png',
#     dpi=300
# )

# plt.show()


# # ============================================================
# # FINAL
# # ============================================================

# print()
# print("=" * 72)
# print("ANÁLISE CONCLUÍDA")
# print("=" * 72)

# print(
#     "Arquivo gerado: joint_velocity_profiles.png"
# )

# print()
# print(
#     "OBSERVAÇÃO METODOLÓGICA:"
# )

# print(
#     "O Jerk RMS foi calculado a partir da velocidade "
#     "articular reamostrada em uma grade temporal uniforme "
#     "e suavizada antes das diferenciações numéricas."
# )

# print(
#     "A quantidade de mensagens /command não é interpretada "
#     "automaticamente como quantidade de trajetórias físicas."
# )











# #!/usr/bin/env python3
# import rosbag
# import numpy as np
# import matplotlib.pyplot as plt

# arquivo_bag = 'ENSAIO GRASP/ensaio_grasping2.bag'

# print(f"--- INICIANDO ANÁLISE CINÉTICA: {arquivo_bag} ---\n")

# try:
#     bag = rosbag.Bag(arquivo_bag)
# except Exception as e:
#     print(f"Erro ao abrir o bag: {e}")
#     exit()

# tempos = []
# velocidades = []
# tempos_comandos = []

# # Lendo os tópicos
# for topic, msg, t in bag.read_messages(topics=['/ur5/joint_states', '/ur5/eff_joint_traj_controller/command']):
#     if topic == '/ur5/joint_states':
#         # Pega as velocidades das 6 juntas do braço (ignorando a garra se houver erro de indexação)
#         if len(msg.velocity) >= 6:
#             tempos.append(msg.header.stamp.to_sec())
#             velocidades.append(msg.velocity[:6])
            
#     elif topic == '/ur5/eff_joint_traj_controller/command':
#         tempos_comandos.append(t.to_sec())

# bag.close()

# # 1. ANÁLISE DE TEMPO DE EXECUÇÃO AUTÔNOMA
# if len(tempos_comandos) > 0:
#     tempo_inicio = tempos_comandos[0]
#     tempo_fim = tempos_comandos[-1]
#     duracao_movimentos = tempo_fim - tempo_inicio
#     print(f"[Tabela 4] Duração da janela de execução autônoma: {duracao_movimentos:.2f} segundos")
#     print(f"           Quantidade de trajetórias executadas: {len(tempos_comandos)}")

# # 2. CÁLCULO DE JERK (SUAVIDADE CINÉTICA)
# tempos = np.array(tempos)
# velocidades = np.array(velocidades)

# # Garante que os tempos começam do zero para facilitar o cálculo
# tempos = tempos - tempos[0]

# # Calcula dt (variação de tempo)
# dt = np.diff(tempos)
# # Evita divisão por zero caso mensagens tenham timestamps idênticos
# indices_validos = dt > 0.001 
# dt = dt[indices_validos]

# # Aceleração = d(Velocidade) / dt
# aceleracoes = np.diff(velocidades, axis=0)[indices_validos] / dt[:, np.newaxis]

# # Jerk = d(Aceleração) / dt
# dt_acc = dt[1:] # Ajusta o tamanho do array de tempo
# jerk = np.diff(aceleracoes, axis=0) / dt_acc[:, np.newaxis]

# # Calcula o RMS (Root Mean Square) do Jerk para todas as juntas
# jerk_rms = np.sqrt(np.mean(jerk**2, axis=0))
# jerk_medio_global = np.mean(jerk_rms)

# print(f"\n[Tabela 4] Análise de Suavidade Cinética (Jerk Médio RMS):")
# print(f" -> Jerk Global (Média do Braço): {jerk_medio_global:.4f} rad/s³")
# print(" -> Detalhamento por Junta:")
# juntas_nomes = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']
# for nome, j in zip(juntas_nomes, jerk_rms):
#     print(f"    - {nome}: {j:.4f} rad/s³")

# # 3. GERAÇÃO DO GRÁFICO PARA O ARTIGO (Opcional, mas recomendado)
# # Plota a velocidade da junta Base (Shoulder Pan) para provar que a curva é suave
# plt.figure(figsize=(10, 5))
# plt.plot(tempos, velocidades[:, 0], label='Shoulder Pan (Base)', color='blue', linewidth=2)
# plt.plot(tempos, velocidades[:, 1], label='Shoulder Lift', color='green', linewidth=2)
# plt.title('Perfil de Velocidade das Juntas (Comprovação da Interpolação Quíntupla)')
# plt.xlabel('Tempo (s)')
# plt.ylabel('Velocidade (rad/s)')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('ENSAIO GRASP/grafico_velocidade_juntas.png', dpi=300)
# print("\n[+] Gráfico de velocidades salvo como 'grafico_velocidade_juntas.png'. Você pode usar isso no artigo!")
