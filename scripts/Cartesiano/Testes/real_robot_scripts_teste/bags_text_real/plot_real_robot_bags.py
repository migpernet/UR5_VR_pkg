#!/usr/bin/env python3
# Arquivo: plot_real_robot_bags.py
# Foco: Validação do amortecedor dinâmico e latência da rede física (Hardware UR5)

import rosbag
import matplotlib.pyplot as plt

def plot_real_streaming_90hz(bag_path):
    print(f"Lendo o arquivo: {bag_path}")
    bag = rosbag.Bag(bag_path)

    time_actual, pos_actual = [], []
    time_cmd, pos_cmd = [], []

    target_joint = "wrist_3_joint"
    start_time = None

    # Tópicos oficiais do hardware
    topics_to_read = ['/joint_states', '/scaled_pos_joint_traj_controller/command']

    for topic, msg, t in bag.read_messages(topics=topics_to_read):
        if start_time is None:
            start_time = t.to_sec()
            
        current_time = t.to_sec() - start_time

        if topic == '/joint_states':
            if target_joint in msg.name:
                idx = msg.name.index(target_joint)
                time_actual.append(current_time)
                pos_actual.append(msg.position[idx])
                
        elif topic == '/scaled_pos_joint_traj_controller/command':
            if target_joint in msg.joint_names:
                idx = msg.joint_names.index(target_joint)
                time_cmd.append(current_time)
                pos_cmd.append(msg.points[0].positions[idx])

    bag.close()

    # --- Plotagem com Qualidade Acadêmica ---
    plt.figure(figsize=(10, 5))
    
    # Linha de comando em azul tracejado
    plt.plot(time_cmd, pos_cmd, label='Comando (VR Streaming 90Hz)', color='blue', linestyle='--', linewidth=2)
    # Resposta física (encoders) em verde escuro
    plt.plot(time_actual, pos_actual, label='Resposta Física (Encoders UR5)', color='green', linewidth=2)
    
    plt.title('Resposta Dinâmica do Hardware a 90Hz (wrist_3_joint)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (segundos)', fontsize=12)
    plt.ylabel('Posição Articular (radianos)', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Destaca a diferença/latência visualmente
    plt.fill_between(time_actual, pos_actual, min(pos_actual)-0.01, color='green', alpha=0.05)

    plt.tight_layout()
    plt.savefig('real_robot_streaming_plot.png', dpi=300)
    print("Gráfico salvo como 'real_robot_streaming_plot.png'. Feche a janela para encerrar.")
    plt.show()

if __name__ == '__main__':
    # Certifique-se de que o nome do bag corresponde ao que você gravou
    plot_real_streaming_90hz('real_log_streaming_90hz.bag')