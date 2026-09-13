rosbag record -O real_log_action_server.bag \
    /joint_states \
    /scaled_pos_joint_traj_controller/follow_joint_trajectory/goal \
    /scaled_pos_joint_traj_controller/follow_joint_trajectory/result