#!/usr/bin/env python3
# Arquivo: plot_gazebo_bags.py
# Foco: Validação da curva de streaming a 90Hz no Gazebo

import rosbag
import matplotlib.pyplot as plt

def plot_streaming_90hz(bag_path):
    print(f"Lendo o arquivo: {bag_path}")
    bag = rosbag.Bag(bag_path)

    # Listas para armazenar os dados de tempo e posição
    time_actual, pos_actual = [], []
    time_cmd, pos_cmd = [], []

    # Nome exato da junta que queremos monitorar
    target_joint = "wrist_3_joint"
    start_time = None

    for topic, msg, t in bag.read_messages(topics=['/ur5/joint_states', '/ur5/eff_joint_traj_controller/command']):
        if start_time is None:
            start_time = t.to_sec()
            
        current_time = t.to_sec() - start_time

        if topic == '/ur5/joint_states':
            if target_joint in msg.name:
                idx = msg.name.index(target_joint)
                time_actual.append(current_time)
                pos_actual.append(msg.position[idx])
                
        elif topic == '/ur5/eff_joint_traj_controller/command':
            if target_joint in msg.joint_names:
                idx = msg.joint_names.index(target_joint)
                # Pega a posição do primeiro ponto da trajetória (já que enviamos 1 ponto por frame no VR)
                time_cmd.append(current_time)
                pos_cmd.append(msg.points[0].positions[idx])

    bag.close()

    # --- Plotagem com Qualidade Acadêmica ---
    plt.figure(figsize=(10, 5))
    plt.plot(time_cmd, pos_cmd, label='Comando (VR Streaming 90Hz)', color='blue', linestyle='--', linewidth=2)
    plt.plot(time_actual, pos_actual, label='Resposta Real (Gazebo)', color='red', linewidth=2)
    
    plt.title('Validação de Streaming a 90Hz - Gazebo (wrist_3_joint)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (segundos)', fontsize=12)
    plt.ylabel('Posição Articular (radianos)', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # Salva o gráfico em alta resolução para artigos
    plt.savefig('gazebo_streaming_plot.png', dpi=300)
    print("Gráfico salvo como 'gazebo_streaming_plot.png'. Feche a janela para encerrar.")
    plt.show()

if __name__ == '__main__':
    # Certifique-se de que o nome do bag corresponde ao que você gravou
    plot_streaming_90hz('gazebo_log_streaming_90hz.bag')