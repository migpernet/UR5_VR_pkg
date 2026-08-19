#!/usr/bin/env python3

# versao sem a heurística, corrigindo o calculo da largura do objeto detectado pelo GGCNN e utilizando a melhor solução para a largura entre 
# o Bounding Box e a estimativa da rede neural. A largura do Bounding Box tem prioridade sobre a estimativa da rede neural.
# ESTA VERSÃO UTILIZA O MÉTODO HÍBRIDO (ESTADO DA ARTE):
# 1. Filtro Espacial (TF): Aniquila a mesa baseada na altura Z real em relação ao base_link.
# 2. FloodFill (Topológico): Contorna apenas a peça selecionada, imune a vazamentos.
# tstado em 29/07/2026 às 23:15h
import time
import numpy as np
import argparse
from skimage.draw import circle_perimeter

import torch
import cv2
import tf2_ros
import tf2_geometry_msgs

import rospy
import rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float32
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
import math

from models.ggcnn import GGCNN 

class ggcnn_grasping(object):
    def __init__(self, args):
        rospy.init_node('ggcnn_detection')

        self.args = args
        self.bridge = CvBridge()
        self.latest_depth_message = None
        self.color_img = None
        
        rospack = rospkg.RosPack()
        Home = rospack.get_path('ggcnn_pkg')
        MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
        self.model = GGCNN()
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        self.model.eval()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
        self.FOV = rospy.get_param("/GGCNN/FOV", 60)
        self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
        if self.args.real:
            self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
        else:
            self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

        self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
        self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
        self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
        self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
        self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
        self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
        self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
        self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

        self.grasping_point = []
        self.depth_image_shot = None
        
        # VARIÁVEL DE INTENÇÃO DO VR
        self.unity_target_base_link = None
        
        camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
        K = camera_info_msg.K
        self.fx = K[0]
        self.cx = K[2]
        self.fy = K[4]
        self.cy = K[5]

        # Os Subscribers do ROS
        rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
        rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

    # ==================================================================
    # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
    # ==================================================================
    def intention_callback(self, msg):
        correct_x = msg.z   
        correct_y = -msg.y  
        correct_z = msg.x   
        
        self.unity_target_base_link = [correct_x, correct_y, correct_z]
        rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

    def get_depth_callback(self, depth_message):
        self.latest_depth_message = depth_message

    def image_callback(self, color_msg):
        self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

    def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
        dx = width / 2
        dy = height / 2
        rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
        rect = rect @ R.T
        rect[:, 0] += x
        rect[:, 1] += y
        rect = rect.astype(np.int32)
        cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
        return img

    def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
        if np.max(map_array) > np.min(map_array):
            normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
        else:
            normalized_map = np.zeros_like(map_array, dtype=np.float32)
        normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
        normalized_map = np.ascontiguousarray(normalized_map)
        colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
        return colorized_map

    def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
        pos_img = self._normalize_and_colorize_map(pos_out)
        ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
        width_img = self._normalize_and_colorize_map(width_out)
        qual_img = self._normalize_and_colorize_map(qual_out)
        
        qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
        rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
        
        valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        qual_img[rr[valid_indices], cc[valid_indices]] = 255
        return pos_img, ang_img, width_img, qual_img

    def depth_process_ggcnn(self):
        self.measured_width_px = None 
        depth_message = self.latest_depth_message
        if depth_message is None or self.color_img is None:
            return

        depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
        depth = depth.astype(np.float32)  
        depth_copy_for_point_depth = depth.copy()
        
        height_res, width_res = depth.shape
        
        offset_x = (width_res - self.crop_size)//2
        offset_y = 0
        depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
        depth_crop = depth_crop.copy()
        
        depth_nan = np.isnan(depth_crop)
        depth_crop[depth_nan] = 0

        mask = (depth_crop == 0).astype(np.uint8)
        depth_scale = np.abs(depth_crop).max()
        depth_crop = depth_crop.astype(np.float32) / depth_scale 
        depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
        depth_crop = depth_crop[1:-1, 1:-1]
        depth_crop = depth_crop * depth_scale

        depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)

        depth_crop = depth_crop/1000.0
        depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
        depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
        self.model.eval() 
        with torch.no_grad(): 
            pred_out = self.model(depth_tensor)  
        
        points_out = pred_out[0].squeeze().cpu().numpy()
        cos_out = pred_out[1].squeeze().cpu().numpy()
        sin_out = pred_out[2].squeeze().cpu().numpy()
        ang_out = np.arctan2(sin_out, cos_out) / 2.0  
        width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
        pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
        pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
        ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
        width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
        mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
            try:
                pt_stamped = tf2_geometry_msgs.PointStamped()
                pt_stamped.header.frame_id = "base_link"
                pt_stamped.point.x = self.unity_target_base_link[0]
                pt_stamped.point.y = self.unity_target_base_link[1]
                pt_stamped.point.z = self.unity_target_base_link[2]
                
                cam_frame = depth_message.header.frame_id
                pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
                if pt_cam.point.z > 0:
                    u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
                    v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
                    cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) 
                    
                    u_crop = u - offset_x
                    v_crop = v - offset_y
                    
                    if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
                        
                        crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size].copy()
                        
                        # ========================================================
                        # ESTÁGIO 1: A MURALHA ESPACIAL (Filtragem por Transformação Z)
                        # ========================================================
                        # Obter a transformação da câmera para o base_link neste exato milissegundo
                        cam_to_base_tf = self.tf_buffer.lookup_transform("base_link", cam_frame, rospy.Time(0), rospy.Duration(0.1))
                        
                        # Matriz de extração eficiente usando numpy (Transformação em Lote)
                        # Cria matrizes de coordenadas X e Y da câmera para todo o crop
                        vs, us = np.indices((self.crop_size, self.crop_size))
                        us_global = us + offset_x
                        vs_global = vs + offset_y
                        
                        # Converte Pinhole para 3D (Câmera)
                        Z_cam = crop_depth_mm / 1000.0  # Converte mm para metros
                        X_cam = (us_global - self.cx) * Z_cam / self.fx
                        Y_cam = (vs_global - self.cy) * Z_cam / self.fy
                        
                        # Achata as matrizes para aplicar TF
                        valid_mask = Z_cam > 0
                        X_flat = X_cam[valid_mask]
                        Y_flat = Y_cam[valid_mask]
                        Z_flat = Z_cam[valid_mask]
                        
                        if len(X_flat) > 0:
                            # Converte a rotação do TF para matriz matemática
                            quat = [cam_to_base_tf.transform.rotation.x, cam_to_base_tf.transform.rotation.y, 
                                    cam_to_base_tf.transform.rotation.z, cam_to_base_tf.transform.rotation.w]
                            
                            # Biblioteca manual de rotação baseada em quaternions para não depender de pacotes externos pesados
                            qx, qy, qz, qw = quat
                            R_matrix = np.array([
                                [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
                                [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
                                [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                            ])
                            
                            trans_vec = np.array([cam_to_base_tf.transform.translation.x, 
                                                  cam_to_base_tf.transform.translation.y, 
                                                  cam_to_base_tf.transform.translation.z])
                            
                            # Aplica a rotação e translação (do ref da câmera para o ref da base do robô)
                            points_cam = np.vstack((X_flat, Y_flat, Z_flat))
                            points_base = R_matrix @ points_cam + trans_vec[:, np.newaxis]
                            
                            # Extrai apenas as alturas (Z) no mundo real do robô
                            Z_base = points_base[2, :]
                            
                            # REGRAS DA MURALHA: Tudo abaixo de 1.5 cm de altura real é apagado (vira buraco negro = 0)
                            # Z da mesa é sempre 0 no Gazebo. 0.015m garante matar a mesa e ruídos de chão.
                            MESA_CUTOFF_M = 0.005  
                            
                            # Remapeia o vetor achatado de volta para o crop
                            Z_base_map = np.zeros_like(Z_cam)
                            Z_base_map[valid_mask] = Z_base
                            
                            # Apaga a mesa da matriz de profundidade em mm
                            crop_depth_mm[Z_base_map < MESA_CUTOFF_M] = 0

                        # ========================================================
                        # ESTÁGIO 2: A ÁGUA TOPOLÓGICA (FloodFill na peça isolada)
                        # ========================================================
                        seed_u, seed_v = int(u_crop), int(v_crop)
                        if crop_depth_mm[seed_v, seed_u] <= 0:
                            c_min_u, c_max_u = max(0, seed_u - 5), min(self.crop_size, seed_u + 5)
                            c_min_v, c_max_v = max(0, seed_v - 5), min(self.crop_size, seed_v + 5)
                            found = False
                            for i in range(c_min_v, c_max_v):
                                for j in range(c_min_u, c_max_u):
                                    if crop_depth_mm[i, j] > 0:
                                        seed_u, seed_v = j, i
                                        found = True
                                        break
                                if found: break
                        
                        if crop_depth_mm[seed_v, seed_u] <= 0:
                            # Se mesmo buscando ao redor não achar peça (usuário clicou onde a mesa foi apagada)
                            cv2.circle(mask_2d, (int(u_crop), int(v_crop)), 45, 1.0, -1)
                            rospy.logwarn_throttle(2.0, "[GGCNN] Intenção no Vazio/Mesa. Abortando Bounding Box.")
                        else:
                            # A mesa já foi deletada. Podemos subir a tolerância do FloodFill sem medo de vazamento!
                            lo_diff = 15.0 # Alta tolerância para absorver ruído do sensor na superfície da peça
                            up_diff = 15.0 
                            
                            flood_mask = np.zeros((self.crop_size + 2, self.crop_size + 2), dtype=np.uint8)
                            flags = 8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
                            
                            cv2.floodFill(crop_depth_mm, flood_mask, (seed_u, seed_v), 255, lo_diff, up_diff, flags)
                            
                            binary_mask = flood_mask[1:-1, 1:-1]
                            
                            # Fechamento morfológico maciço para peças complexas
                            kernel = np.ones((7,7), np.uint8)
                            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                            
                            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            target_contour = None
                            min_dist = float('inf')
                            
                            for cnt in contours:
                                dist = cv2.pointPolygonTest(cnt, (int(u_crop), int(v_crop)), True)
                                if dist >= 0:
                                    target_contour = cnt
                                    break
                                elif abs(dist) < min_dist:
                                    min_dist = abs(dist)
                                    target_contour = cnt 
                            
                            if target_contour is not None and len(target_contour) >= 3:
                                rect = cv2.minAreaRect(target_contour)
                                box = cv2.boxPoints(rect)
                                box = np.int32(box) 
                                
                                cv2.drawContours(mask_2d, [box], 0, 1.0, -1)
                                
                                box_full = box + np.array([offset_x, offset_y])
                                cv2.drawContours(debug_view, [box_full], 0, (255, 0, 0), 2)

                                self.measured_width_px = min(rect[1][0], rect[1][1])
                                rospy.loginfo_throttle(1.0, "[GGCNN] Bounding Box Híbrido aplicado (TF + FloodFill)!")
                            else:
                                cv2.circle(mask_2d, (int(u_crop), int(v_crop)), 45, 1.0, -1)
                                rospy.logwarn_throttle(2.0, f"[GGCNN] Contorno perdido. Usando mascara circular.")

                        pos_out_filtered = pos_out_filtered * mask_2d

                    else:
                        cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
                        cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
                        rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

        cv2.imshow("Debug: Projecao de Intencao", debug_view)
        cv2.imshow("Debug: Mascara 300x300", mask_2d)
        cv2.waitKey(1)

        try:
            transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
            ROBOT_Z = transform_stamped.transform.translation.z
        except:
            ROBOT_Z = 0.0
        
        max_idx = np.argmax(pos_out_filtered)
        best_pixel = np.unravel_index(max_idx, pos_out_filtered.shape)
        
        max_pixel = np.array(best_pixel)
        grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]
        
        rospy.loginfo_throttle(1.0, f"[GGCNN] Qualidade (Max Global): {grasp_quality:.3f}")
        
        if grasp_quality < 0.001:
            return

        self.best_y, self.best_x = max_pixel.astype(int)
        ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   

        if hasattr(self, 'measured_width_px') and self.measured_width_px is not None:
            width_px = self.measured_width_px
            rospy.loginfo_throttle(1.0, f"[LARGURA] Override! Usando tamanho real do Bounding Box: {width_px:.1f}px")
        else:
            width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
            rospy.loginfo_throttle(1.0, "[LARGURA] Usando estimativa bruta da GGCNN")
            
        reescaled_height = int(max_pixel[0]) 
        reescaled_width = int(offset_x + max_pixel[1])
        max_pixel_reescaled = [reescaled_height, reescaled_width]
        point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

        g_width = width_px 
        
        width_m = (width_px * point_depth) / (self.fx * 1000.0)

        if not np.isnan(point_depth):
            x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
            y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
            grasping_point = [x, y, point_depth] 

            self.ang_out = ang_out
            self.width_out = width_out
            self.points_out = points_out
            self.depth_message_ggcnn = depth_message
            self.depth_crop = depth_crop
            self.ang = ang 
            self.width_px = width_px
            self.max_pixel = max_pixel
            self.max_pixel_reescaled = max_pixel_reescaled
            self.g_width = g_width
            self.width_m = width_m
            self.point_depth = point_depth
            self.grasping_point = grasping_point
            self.qual_out = grasp_quality   
            self.pos_out_filtered = pos_out_filtered

    def publish_images(self):
        if not hasattr(self, 'points_out') or self.points_out is None:
            return
        
        pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
            self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
        )
        pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
        pos_msg.header = self.depth_message_ggcnn.header
        self.grasp_pub.publish(pos_msg)

        ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
        ang_msg.header = self.depth_message_ggcnn.header
        self.ang_pub.publish(ang_msg)

        width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
        width_msg.header = self.depth_message_ggcnn.header
        self.width_pub.publish(width_msg)
        
        qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
        qual_msg.header = self.depth_message_ggcnn.header
        self.depth_pub.publish(qual_msg)

    def publish_data_to_robot(self):
        if not hasattr(self, 'grasping_point') or not self.grasping_point:
            return

        cmd_msg = Float32MultiArray()
        cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
        self.cmd_pub.publish(cmd_msg)
        
        grasp_transform = TransformStamped()
        grasp_transform.header.stamp = rospy.Time.now()
        grasp_transform.header.frame_id = "camera_depth_optical_frame"
        grasp_transform.child_frame_id = "object_detected"
        grasp_transform.transform.translation.x = cmd_msg.data[0]
        grasp_transform.transform.translation.y = cmd_msg.data[1]
        grasp_transform.transform.translation.z = cmd_msg.data[2]
        
        q = quaternion_from_euler(0.0, 0.0, 0.0) 
        grasp_transform.transform.rotation.x = q[0]
        grasp_transform.transform.rotation.y = q[1]
        grasp_transform.transform.rotation.z = q[2]
        grasp_transform.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(grasp_transform)

    def get_transform_between_frames(self, target_frame, source_frame):
        if not hasattr(self, 'grasping_point') or not self.grasping_point:
            return None

        try:
            cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.2))
            
            point_cam = tf2_geometry_msgs.PointStamped()
            point_cam.header.frame_id = "camera_depth_optical_frame"
            point_cam.header.stamp = rospy.Time(0)
            point_cam.point.x = self.grasping_point[0] / 1000.0
            point_cam.point.y = self.grasping_point[1] / 1000.0
            point_cam.point.z = self.grasping_point[2] / 1000.0
            
            point_base = self.tf_buffer.transform(point_cam, target_frame, rospy.Duration(0.2))
            x = point_base.point.x
            y = point_base.point.y
            z = point_base.point.z
            
            q_cam = [
                cam_transform.transform.rotation.x,
                cam_transform.transform.rotation.y,
                cam_transform.transform.rotation.z,
                cam_transform.transform.rotation.w
            ]
            euler_cam = euler_from_quaternion(q_cam)
            camera_yaw = euler_cam[2]
            
            final_roll = math.pi 
            final_pitch = 0.0
            OFFSET_YAW = 1.5708    
            final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
            quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)
            
            unity_pose = PoseStamped()
            unity_pose.header.stamp = rospy.Time.now()
            unity_pose.header.frame_id = target_frame 
            unity_pose.pose.position.x = x
            unity_pose.pose.position.y = y
            unity_pose.pose.position.z = z
            unity_pose.pose.orientation.x = quat[0]
            unity_pose.pose.orientation.y = quat[1]
            unity_pose.pose.orientation.z = quat[2]
            unity_pose.pose.orientation.w = quat[3]

            self.unity_pose_pub.publish(unity_pose)
            self.unity_width_pub.publish(self.width_m)

            cmd_msg_grasp = Float32MultiArray()
            cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
            self.cmd_pub_grasp.publish(cmd_msg_grasp)

            self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
            return cam_transform
            
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[GHOST GRIPPER] Aguardando sincronia de TF: {e}")
            return None
            
    def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transform_stamped = TransformStamped()
        static_transform_stamped.header.stamp = rospy.Time.now()
        static_transform_stamped.header.frame_id = parent_frame
        static_transform_stamped.child_frame_id = child_frame
        static_transform_stamped.transform.translation.x = x
        static_transform_stamped.transform.translation.y = y
        static_transform_stamped.transform.translation.z = z
        
        static_transform_stamped.transform.rotation.x = quat[0]
        static_transform_stamped.transform.rotation.y = quat[1]
        static_transform_stamped.transform.rotation.z = quat[2]
        static_transform_stamped.transform.rotation.w = quat[3]
        tf_broadcaster.sendTransform(static_transform_stamped)

def parse_args():
    parser = argparse.ArgumentParser(description='GGCNN grasping')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    grasp_detection = ggcnn_grasping(args)
    rospy.sleep(1.0)
    print("Iniciando processo GGCNN...")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        grasp_detection.depth_process_ggcnn()
        grasp_detection.publish_images()
        grasp_detection.publish_data_to_robot()
        grasp_detection.get_transform_between_frames("base_link", "object_detected")
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
















# #!/usr/bin/env python3

# # versao sem a heurística, corrigindo o calculo da largura do objeto detectado pelo GGCNN e utilizando a melhor solução para a largura entre 
# # o Bounding Box e a estimativa da rede neural. A largura do Bounding Box tem prioridade sobre a estimativa da rede neural.
# # esta versão também automatiza a criação do bounding box dinâmico utilizando Segmentação Topológica (FloodFill - "derramando água vitual") a partir do ponto de intenção do VR.
# # imune à inclinação da mesa, baseado na profundidade do ponto de intenção do VR.
# # Funionou de forma satisfatória.
# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         correct_x = msg.z   
#         correct_y = -msg.y  
#         correct_z = msg.x   
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
        
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         self.measured_width_px = None # Zera a leitura anterior a cada frame
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)

#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) 
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
                        
#                         # ========================================================
#                         # SEGMENTAÇÃO TOPOLÓGICA (FloodFill) - Imune a Mesas Inclinadas
#                         # ========================================================
#                         crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size].copy()
                        
#                         # 1. Encontrar uma semente válida (caso o pixel exato seja um "buraco negro")
#                         seed_u, seed_v = int(u_crop), int(v_crop)
#                         if crop_depth_mm[seed_v, seed_u] <= 0:
#                             c_min_u, c_max_u = max(0, seed_u - 5), min(self.crop_size, seed_u + 5)
#                             c_min_v, c_max_v = max(0, seed_v - 5), min(self.crop_size, seed_v + 5)
#                             found = False
#                             for i in range(c_min_v, c_max_v):
#                                 for j in range(c_min_u, c_max_u):
#                                     if crop_depth_mm[i, j] > 0:
#                                         seed_u, seed_v = j, i
#                                         found = True
#                                         break
#                                 if found: break
                        
#                         if crop_depth_mm[seed_v, seed_u] <= 0:
#                             # Fallback extremo se o clique for totalmente no vazio
#                             cv2.circle(mask_2d, (int(u_crop), int(v_crop)), 95, 1.0, -1)
#                         else:
#                             # 2. Algoritmo de Crescimento de Região (Topológico)
#                             # A variação entre pixels na superfície lisa do objeto é minúscula (<1mm)
#                             # O degrau caindo para a mesa é abrupto (vários mm de diferença)
#                             lo_diff = 3.0 # Limite de degrau descendo
#                             up_diff = 3.0 # Limite de degrau subindo
                            
#                             flood_mask = np.zeros((self.crop_size + 2, self.crop_size + 2), dtype=np.uint8)
#                             flags = 8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
                            
#                             # Derrama "água virtual" no ponto de clique. Ela se espalha até bater num degrau maior que 3mm.
#                             cv2.floodFill(crop_depth_mm, flood_mask, (seed_u, seed_v), 255, lo_diff, up_diff, flags)
                            
#                             binary_mask = flood_mask[1:-1, 1:-1]
                            
#                             # Fecha micro-buracos de ruído na superfície
#                             kernel = np.ones((5,5), np.uint8)
#                             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                            
#                             # 3. Encontra os Contornos 
#                             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
#                             target_contour = None
#                             min_dist = float('inf')
                            
#                             for cnt in contours:
#                                 dist = cv2.pointPolygonTest(cnt, (int(u_crop), int(v_crop)), True)
#                                 if dist >= 0:
#                                     target_contour = cnt
#                                     break
#                                 elif abs(dist) < min_dist:
#                                     min_dist = abs(dist)
#                                     target_contour = cnt 
                            
#                             if target_contour is not None and len(target_contour) >= 3:
#                                 rect = cv2.minAreaRect(target_contour)
#                                 box = cv2.boxPoints(rect)
#                                 box = np.int32(box) 
                                
#                                 cv2.drawContours(mask_2d, [box], 0, 1.0, -1)
                                
#                                 box_full = box + np.array([offset_x, offset_y])
#                                 cv2.drawContours(debug_view, [box_full], 0, (255, 0, 0), 2)

#                                 self.measured_width_px = min(rect[1][0], rect[1][1])
#                                 rospy.loginfo_throttle(1.0, "[GGCNN] Bounding Box Topologico aplicado!")
#                             else:
#                                 cv2.circle(mask_2d, (int(u_crop), int(v_crop)), 45, 1.0, -1)
#                                 rospy.logwarn_throttle(2.0, f"[GGCNN] Contorno perdido. Usando mascara circular.")

#                         pos_out_filtered = pos_out_filtered * mask_2d

#                     else:
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         max_idx = np.argmax(pos_out_filtered)
#         best_pixel = np.unravel_index(max_idx, pos_out_filtered.shape)
        
#         max_pixel = np.array(best_pixel)
#         grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]
        
#         rospy.loginfo_throttle(1.0, f"[GGCNN] Qualidade (Max Global): {grasp_quality:.3f}")
        
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   

#         if hasattr(self, 'measured_width_px') and self.measured_width_px is not None:
#             width_px = self.measured_width_px
#             rospy.loginfo_throttle(1.0, f"[LARGURA] Override! Usando tamanho real do Bounding Box: {width_px:.1f}px")
#         else:
#             width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
#             rospy.loginfo_throttle(1.0, "[LARGURA] Usando estimativa bruta da GGCNN")
            
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = width_px 
        
#         width_m = (width_px * point_depth) / (self.fx * 1000.0)

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
        
#         q = quaternion_from_euler(0.0, 0.0, 0.0) 
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return None

#         try:
#             cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.2))
            
#             point_cam = tf2_geometry_msgs.PointStamped()
#             point_cam.header.frame_id = "camera_depth_optical_frame"
#             point_cam.header.stamp = rospy.Time(0)
#             point_cam.point.x = self.grasping_point[0] / 1000.0
#             point_cam.point.y = self.grasping_point[1] / 1000.0
#             point_cam.point.z = self.grasping_point[2] / 1000.0
            
#             point_base = self.tf_buffer.transform(point_cam, target_frame, rospy.Duration(0.2))
#             x = point_base.point.x
#             y = point_base.point.y
#             z = point_base.point.z
            
#             q_cam = [
#                 cam_transform.transform.rotation.x,
#                 cam_transform.transform.rotation.y,
#                 cam_transform.transform.rotation.z,
#                 cam_transform.transform.rotation.w
#             ]
#             euler_cam = euler_from_quaternion(q_cam)
#             camera_yaw = euler_cam[2]
            
#             final_roll = math.pi 
#             final_pitch = 0.0
#             OFFSET_YAW = 1.5708    
#             final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
#             quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
#             return cam_transform
            
#         except Exception as e:
#             rospy.logwarn_throttle(2.0, f"[GHOST GRIPPER] Aguardando sincronia de TF: {e}")
#             return None
            
#     def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
        
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass















# #!/usr/bin/env python3

# # versao sem a heurística, corrigindo o calculo da largura do objeto detectado pelo GGCNN e utilizando a melhor solução para a largura entre 
# # o Bounding Box e a estimativa da rede neural. A largura do Bounding Box tem prioridade sobre a estimativa da rede neural.
# # esta versão automatiza a criação do bounding box dinâmico utilizando a Janela de Isolamento Local (Sub-Crop) 
# # imune à inclinação da mesa, baseado na profundidade do ponto de intenção do VR.
# # Cria uma janela de isolamento local (sub-crop) de 100x100 pixels em torno do ponto de intenção do VR. Não especificamente em torno do objeto.
# # Data de alteração: 29/07/2026 às 18:44h

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         correct_x = msg.z   
#         correct_y = -msg.y  
#         correct_z = msg.x   
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
        
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         self.measured_width_px = None # Zera a leitura anterior a cada frame
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)

#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) 
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
                        
#                         # ========================================================
#                         # NOVA ABORDAGEM: JANELA DE ISOLAMENTO LOCAL (Sub-Crop)
#                         # Imune à inclinação da mesa, ruídos de borda e vazamentos.
#                         # ========================================================
#                         crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
                        
#                         # 1. Definir uma janela de isolamento ao redor do clique (120x120 pixels)
#                         # A garra abre no máximo 14cm. 120px é mais que suficiente e corta o resto da mesa inclinada!
#                         BOX_RADIUS = 60
#                         c_min_u = max(0, int(u_crop) - BOX_RADIUS)
#                         c_max_u = min(self.crop_size, int(u_crop) + BOX_RADIUS)
#                         c_min_v = max(0, int(v_crop) - BOX_RADIUS)
#                         c_max_v = min(self.crop_size, int(v_crop) + BOX_RADIUS)
                        
#                         local_depth = crop_depth_mm[c_min_v:c_max_v, c_min_u:c_max_u]
                        
#                         # 2. Descobrir a profundidade exata do clique (mini-janela de 10x10 para evitar buracos negros)
#                         mini_u_min = max(0, int(u_crop) - 5)
#                         mini_u_max = min(self.crop_size, int(u_crop) + 5)
#                         mini_v_min = max(0, int(v_crop) - 5)
#                         mini_v_max = min(self.crop_size, int(v_crop) + 5)
#                         mini_window = crop_depth_mm[mini_v_min:mini_v_max, mini_u_min:mini_u_max]
#                         valid_mini = mini_window[mini_window > 0]
                        
#                         if len(valid_mini) > 0:
#                             click_depth = np.percentile(valid_mini, 10) # Âncora no topo da peça
#                         else:
#                             click_depth = 0.0

#                         if click_depth <= 0:
#                             cv2.circle(mask_2d, (int(u_crop), int(v_crop)), 45, 1.0, -1)
#                         else:
#                             # 3. Descobrir a profundidade da mesa APENAS dentro da Janela Local
#                             valid_local = local_depth[local_depth > 0]
#                             if len(valid_local) > 0:
#                                 table_depth = np.percentile(valid_local, 95)
#                             else:
#                                 table_depth = click_depth + 50.0
                                
#                             # Se a diferença for minúscula (< 5mm), o usuário clicou na mesa por engano
#                             if abs(table_depth - click_depth) < 5.0:
#                                 rospy.logwarn_throttle(2.0, "[GGCNN] Intencao na mesa. Abortando Bounding Box.")
#                                 cv2.circle(mask_2d, (int(u_crop), int(v_crop)), 45, 1.0, -1)
#                             else:
#                                 # 4. Fatiar a imagem isolando perfeitamente a peça
#                                 max_depth_allowed = table_depth - 4.0  # Trava mecânica de 4mm acima da mesa
#                                 min_depth_allowed = click_depth - 15.0 # Tolera ruídos para cima (sensor)
                                
#                                 local_binary = cv2.inRange(local_depth, min_depth_allowed, max_depth_allowed)
                                
#                                 kernel = np.ones((3,3), np.uint8)
#                                 local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_OPEN, kernel)
                                
#                                 # 5. Encontrar contornos DENTRO da janela local
#                                 contours, _ = cv2.findContours(local_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
#                                 target_contour = None
#                                 min_dist = float('inf')
                                
#                                 # O clique em relação à janela local
#                                 local_click_u = int(u_crop) - c_min_u
#                                 local_click_v = int(v_crop) - c_min_v
                                
#                                 for cnt in contours:
#                                     dist = cv2.pointPolygonTest(cnt, (local_click_u, local_click_v), True)
#                                     if dist >= 0:
#                                         target_contour = cnt
#                                         break
#                                     elif abs(dist) < min_dist:
#                                         min_dist = abs(dist)
#                                         target_contour = cnt
                                
#                                 if target_contour is not None and len(target_contour) >= 3:
#                                     # Deslocar o contorno de volta para as coordenadas reais da imagem (300x300)
#                                     target_contour = target_contour + np.array([[[c_min_u, c_min_v]]])
                                    
#                                     rect = cv2.minAreaRect(target_contour)
#                                     box = cv2.boxPoints(rect)
#                                     box = np.int32(box)
                                    
#                                     cv2.drawContours(mask_2d, [box], 0, 1.0, -1)
                                    
#                                     box_full = box + np.array([offset_x, offset_y])
#                                     cv2.drawContours(debug_view, [box_full], 0, (255, 0, 0), 2)
                                    
#                                     self.measured_width_px = min(rect[1][0], rect[1][1])
#                                     rospy.loginfo_throttle(1.0, "[GGCNN] Bounding Box de Isolamento Local aplicado!")
#                                 else:
#                                     cv2.circle(mask_2d, (int(u_crop), int(v_crop)), 45, 1.0, -1)
#                                     rospy.logwarn_throttle(2.0, "[GGCNN] Contorno perdido na Janela Local.")
                        
#                         pos_out_filtered = pos_out_filtered * mask_2d

#                     else:
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         max_idx = np.argmax(pos_out_filtered)
#         best_pixel = np.unravel_index(max_idx, pos_out_filtered.shape)
        
#         max_pixel = np.array(best_pixel)
#         grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]
        
#         rospy.loginfo_throttle(1.0, f"[GGCNN] Qualidade (Max Global): {grasp_quality:.3f}")
        
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   

#         if hasattr(self, 'measured_width_px') and self.measured_width_px is not None:
#             width_px = self.measured_width_px
#             rospy.loginfo_throttle(1.0, f"[LARGURA] Override! Usando tamanho real do Bounding Box: {width_px:.1f}px")
#         else:
#             width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
#             rospy.loginfo_throttle(1.0, "[LARGURA] Usando estimativa bruta da GGCNN")
            
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = width_px 
        
#         width_m = (width_px * point_depth) / (self.fx * 1000.0)

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
        
#         q = quaternion_from_euler(0.0, 0.0, 0.0) 
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return None

#         try:
#             cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.2))
            
#             point_cam = tf2_geometry_msgs.PointStamped()
#             point_cam.header.frame_id = "camera_depth_optical_frame"
#             point_cam.header.stamp = rospy.Time(0)
#             point_cam.point.x = self.grasping_point[0] / 1000.0
#             point_cam.point.y = self.grasping_point[1] / 1000.0
#             point_cam.point.z = self.grasping_point[2] / 1000.0
            
#             point_base = self.tf_buffer.transform(point_cam, target_frame, rospy.Duration(0.2))
#             x = point_base.point.x
#             y = point_base.point.y
#             z = point_base.point.z
            
#             q_cam = [
#                 cam_transform.transform.rotation.x,
#                 cam_transform.transform.rotation.y,
#                 cam_transform.transform.rotation.z,
#                 cam_transform.transform.rotation.w
#             ]
#             euler_cam = euler_from_quaternion(q_cam)
#             camera_yaw = euler_cam[2]
            
#             final_roll = math.pi 
#             final_pitch = 0.0
#             OFFSET_YAW = 1.5708    
#             final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
#             quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
#             return cam_transform
            
#         except Exception as e:
#             rospy.logwarn_throttle(2.0, f"[GHOST GRIPPER] Aguardando sincronia de TF: {e}")
#             return None
            
#     def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
        
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass



















# #!/usr/bin/env python3

# # versao sem a heurística, corrigindo o calculo da largura do objeto detectado pelo GGCNN e utilizando a melhor solução para a largura entre 
# # o Bounding Box e a estimativa da rede neural. A largura do Bounding Box tem prioridade sobre a estimativa da rede neural.
# # O ajuste de altura de detecção para objetos está manual nesta versão.

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         self.measured_width_px = None # Zera a leitura anterior a cada frame
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         # Tratamento de buracos/NaNs na imagem de profundidade
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # ========================================================
#         # [ESTRATÉGIA DE PERCEPÇÃO] Filtro Gaussiano Espacial
#         # Mitiga o aliasing do VoxelGrid para estabilizar o vetor normal
#         # e forçar o GGCNN a encontrar ângulos ortogonais nas faces.
#         # ========================================================
#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)
#         # ========================================================

#         # Normalização e conversão para o formato da rede
#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG E SEGMENTAÇÃO)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
                        
#                         # ========================================================
#                         # SEGMENTAÇÃO DINÂMICA POR BOUNDING BOX ORIENTADO
#                         # ========================================================
#                         real_u = u_crop + offset_x
#                         real_v = v_crop + offset_y
                        
#                         # --- SOLUÇÃO ANTI-PISCAR 1: Leitura Robusta em Janela ---
#                         min_u = max(0, real_u - 5)
#                         max_u = min(width_res, real_u + 5)
#                         min_v = max(0, real_v - 5)
#                         max_v = min(height_res, real_v + 5)
                        
#                         window = depth_copy_for_point_depth[min_v:max_v, min_u:max_u]
#                         valid_depths = window[~np.isnan(window) & (window > 0)]
                        
#                         if len(valid_depths) > 0:
#                             click_depth = np.median(valid_depths)
#                         else:
#                             click_depth = 0.0

#                         # Se o clique for muito ruidoso, usa o círculo como segurança
#                         if click_depth <= 0:
#                             cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                         else:
#                             # ========================================================
#                             # [AJUSTE] TOLERÂNCIA ASSIMÉTRICA PARA OBJETOS BAIXOS
#                             # ========================================================
#                             TOLERANCE_UP = 30.0   # Permite capturar variações/ruídos no topo da peça
#                             TOLERANCE_DOWN = 5.0  # Limite SUPER RESTRITO para baixo, evitando "vazar" para a mesa
                            
#                             # Isola a área de recorte em milímetros
#                             crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
                            
#                             # Cria a imagem binária filtrando com tolerância assimétrica
#                             binary_mask = cv2.inRange(crop_depth_mm, click_depth - TOLERANCE_UP, click_depth + TOLERANCE_DOWN)
                            
#                             # Filtro Morfológico (Opening) para remover "sujeiras" e pixels isolados da mesa
#                             kernel = np.ones((3,3), np.uint8)
#                             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                            
#                             # Encontra os Contornos da peça isolada
#                             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
#                             target_contour = None
#                             min_dist = float('inf')
                            
#                             # Acha qual contorno pertence ao clique do operador
#                             for cnt in contours:
#                                 dist = cv2.pointPolygonTest(cnt, (u_crop, v_crop), True)
#                                 if dist >= 0:
#                                     target_contour = cnt
#                                     break
#                                 elif abs(dist) < min_dist:
#                                     min_dist = abs(dist)
#                                     target_contour = cnt 
                            
#                             if target_contour is not None and len(target_contour) >= 3:
#                                 # Cria o Bounding Box Orientado exato ao redor da peça
#                                 rect = cv2.minAreaRect(target_contour)
#                                 box = cv2.boxPoints(rect)
#                                 box = np.int32(box) # np.int32 evita bugs em novas versões do Numpy
                                
#                                 # Pinta o Bounding Box de branco na máscara
#                                 cv2.drawContours(mask_2d, [box], 0, 1.0, -1)
                                
#                                 # Desenha o Bounding Box AZUL na visão de Debug
#                                 box_full = box + np.array([offset_x, offset_y])
#                                 cv2.drawContours(debug_view, [box_full], 0, (255, 0, 0), 2)

#                                 # ========================================================
#                                 # EXTRAÇÃO GEOMÉTRICA DA LARGURA REAL
#                                 # ========================================================
#                                 # O minAreaRect retorna ((center_x, center_y), (width, height), angle)
#                                 # A menor dimensão do retângulo é sempre a largura real de preensão da peça!
#                                 self.measured_width_px = min(rect[1][0], rect[1][1])
#                                 # ========================================================

#                                 rospy.loginfo_throttle(1.0, "[GGCNN] Bounding Box Dinamico aplicado!")
#                             else:
#                                 # Se o TF calcular o ponto ligeiramente fora do crop devido a vibração do robô,
#                                 # não abortamos! Forçamos uma máscara esférica no centro da imagem (150, 150)
#                                 cv2.circle(mask_2d, (150, 150), 95, 1.0, -1)
#                                 pos_out_filtered = pos_out_filtered * mask_2d
#                                 rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: Forçando máscara no centro.")

#                         # Multiplica o mapa de calor pela nova máscara matemática
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         # ========================================================

#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         # ==========================================================
#         # [ESTRATÉGIA DE DECISÃO] Máximo Global Simples (Sem Heurística)
#         # ==========================================================
#         # Extrai de forma absoluta o pixel com a maior qualidade de preensão no mapa filtrado.
#         max_idx = np.argmax(pos_out_filtered)
#         best_pixel = np.unravel_index(max_idx, pos_out_filtered.shape)
        
#         max_pixel = np.array(best_pixel)
#         grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]
        
#         rospy.loginfo_throttle(1.0, f"[GGCNN] Qualidade (Max Global): {grasp_quality:.3f}")
#         # ==========================================================
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   

#         # ==========================================================
#         # CORREÇÃO GEOMÉTRICA 2: Cálculo Final de Abertura (Pinhole)
#         # ==========================================================
#         # Se a nossa segmentação dinâmica encontrou a peça perfeitamente, ignoramos a IA 
#         # e usamos a medida geométrica real do contorno!
#         if hasattr(self, 'measured_width_px') and self.measured_width_px is not None:
#             width_px = self.measured_width_px
#             rospy.loginfo_throttle(1.0, f"[LARGURA] Override! Usando tamanho real do Bounding Box: {width_px:.1f}px")
#         else:
#             # Caso o Bounding Box falhe, confiamos na estimativa da Rede Neural
#             width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
#             rospy.loginfo_throttle(1.0, "[LARGURA] Usando estimativa bruta da GGCNN")
            
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         # O g_width repassa a largura em pixels para a classe.
#         g_width = width_px 
        
#         # O width_m assume a matemática perfeita baseada na lente da RealSense.
#         width_m = (width_px * point_depth) / (self.fx * 1000.0)
#         # ==========================================================

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         # Publica o TF do objeto na câmera apenas para translação (posição).
#         # Vamos deixar a rotação zerada aqui para não misturar matrizes.
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
        
#         # Rotação neutra. A mágica da orientação vai acontecer no base_link.
#         q = quaternion_from_euler(0.0, 0.0, 0.0) 
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         # Se a rede neural não calculou um ponto válido neste frame, aborta silenciosamente
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return None

#         try:
#             # =================================================================
#             # SOLUÇÃO DEFINITIVA CONTRA O PISCA-PISCA (Transformação Direta)
#             # =================================================================
#             # Em vez de ler o frame "object_detected" (que sofre de Race Condition),
#             # nós lemos a transformação estável da câmera em relação à base_link.
#             # Essa transformação NUNCA falha pois o robô publica as juntas constantemente.
#             cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.2))
            
#             # Criamos um ponto carimbado com as coordenadas locais da câmera.
#             # Lembre-se: grasping_point está em milímetros no seu script, convertemos para metros (/1000.0)
#             point_cam = tf2_geometry_msgs.PointStamped()
#             point_cam.header.frame_id = "camera_depth_optical_frame"
#             point_cam.header.stamp = rospy.Time(0)
#             point_cam.point.x = self.grasping_point[0] / 1000.0
#             point_cam.point.y = self.grasping_point[1] / 1000.0
#             point_cam.point.z = self.grasping_point[2] / 1000.0
            
#             # O ROS faz a matemática vetorial direta transpondo o ponto para a base_link
#             point_base = self.tf_buffer.transform(point_cam, target_frame, rospy.Duration(0.2))
#             x = point_base.point.x
#             y = point_base.point.y
#             z = point_base.point.z
            
#             # Extrai o Yaw da câmera para calcular a orientação ortogonal no plano do mundo
#             q_cam = [
#                 cam_transform.transform.rotation.x,
#                 cam_transform.transform.rotation.y,
#                 cam_transform.transform.rotation.z,
#                 cam_transform.transform.rotation.w
#             ]
#             euler_cam = euler_from_quaternion(q_cam)
#             camera_yaw = euler_cam[2]
            
#             # Mantém a nossa fórmula ortogonal perfeita com o offset corrigido
#             final_roll = math.pi 
#             final_pitch = 0.0
#             OFFSET_YAW = 1.5708    
#             final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
#             quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)
            
#             # Monta a mensagem estável para o Unity
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             # Publica sem NENHUMA perda de pacotes
#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
#             return cam_transform
            
#         except Exception as e:
#             # Caso ocorra um erro de inicialização de nós do ROS no primeiro ciclo, loga suavemente
#             rospy.logwarn_throttle(2.0, f"[GHOST GRIPPER] Aguardando sincronia de TF: {e}")
#             return None
            
#     def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
        
#         # Utiliza o quaternion recebido diretamente
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass






















# #!/usr/bin/env python3

# # versao sem a heurística e corrigindo o calculo da largura do objeto detectado pelo GGCNN.
# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         # Tratamento de buracos/NaNs na imagem de profundidade
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # ========================================================
#         # [ESTRATÉGIA DE PERCEPÇÃO] Filtro Gaussiano Espacial
#         # Mitiga o aliasing do VoxelGrid para estabilizar o vetor normal
#         # e forçar o GGCNN a encontrar ângulos ortogonais nas faces.
#         # ========================================================
#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)
#         # ========================================================

#         # Normalização e conversão para o formato da rede
#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG E SEGMENTAÇÃO)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
                        
#                         # ========================================================
#                         # SEGMENTAÇÃO DINÂMICA POR BOUNDING BOX ORIENTADO
#                         # ========================================================
#                         real_u = u_crop + offset_x
#                         real_v = v_crop + offset_y
                        
#                         # --- SOLUÇÃO ANTI-PISCAR 1: Leitura Robusta em Janela ---
#                         min_u = max(0, real_u - 5)
#                         max_u = min(width_res, real_u + 5)
#                         min_v = max(0, real_v - 5)
#                         max_v = min(height_res, real_v + 5)
                        
#                         window = depth_copy_for_point_depth[min_v:max_v, min_u:max_u]
#                         valid_depths = window[~np.isnan(window) & (window > 0)]
                        
#                         if len(valid_depths) > 0:
#                             click_depth = np.median(valid_depths)
#                         else:
#                             click_depth = 0.0

#                         # Se o clique for muito ruidoso, usa o círculo como segurança
#                         if click_depth <= 0:
#                             cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                         else:
#                             # ========================================================
#                             # [AJUSTE] TOLERÂNCIA ASSIMÉTRICA PARA OBJETOS BAIXOS
#                             # ========================================================
#                             TOLERANCE_UP = 30.0   # Permite capturar variações/ruídos no topo da peça
#                             TOLERANCE_DOWN = 1.0  # Limite SUPER RESTRITO para baixo, evitando "vazar" para a mesa
                            
#                             # Isola a área de recorte em milímetros
#                             crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
                            
#                             # Cria a imagem binária filtrando com tolerância assimétrica
#                             binary_mask = cv2.inRange(crop_depth_mm, click_depth - TOLERANCE_UP, click_depth + TOLERANCE_DOWN)
                            
#                             # Filtro Morfológico (Opening) para remover "sujeiras" e pixels isolados da mesa
#                             kernel = np.ones((3,3), np.uint8)
#                             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                            
#                             # Encontra os Contornos da peça isolada
#                             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
#                             target_contour = None
#                             min_dist = float('inf')
                            
#                             # Acha qual contorno pertence ao clique do operador
#                             for cnt in contours:
#                                 dist = cv2.pointPolygonTest(cnt, (u_crop, v_crop), True)
#                                 if dist >= 0:
#                                     target_contour = cnt
#                                     break
#                                 elif abs(dist) < min_dist:
#                                     min_dist = abs(dist)
#                                     target_contour = cnt 
                            
#                             if target_contour is not None and len(target_contour) >= 3:
#                                 # Cria o Bounding Box Orientado exato ao redor da peça
#                                 rect = cv2.minAreaRect(target_contour)
#                                 box = cv2.boxPoints(rect)
#                                 box = np.int32(box) # np.int32 evita bugs em novas versões do Numpy
                                
#                                 # Pinta o Bounding Box de branco na máscara
#                                 cv2.drawContours(mask_2d, [box], 0, 1.0, -1)
                                
#                                 # Desenha o Bounding Box AZUL na visão de Debug
#                                 box_full = box + np.array([offset_x, offset_y])
#                                 cv2.drawContours(debug_view, [box_full], 0, (255, 0, 0), 2)
#                                 rospy.loginfo_throttle(1.0, "[GGCNN] Bounding Box Dinamico aplicado!")
#                             else:
#                                 # Se o TF calcular o ponto ligeiramente fora do crop devido a vibração do robô,
#                                 # não abortamos! Forçamos uma máscara esférica no centro da imagem (150, 150)
#                                 cv2.circle(mask_2d, (150, 150), 95, 1.0, -1)
#                                 pos_out_filtered = pos_out_filtered * mask_2d
#                                 rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: Forçando máscara no centro.")

#                         # Multiplica o mapa de calor pela nova máscara matemática
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         # ========================================================

#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         # ==========================================================
#         # [ESTRATÉGIA DE DECISÃO] Máximo Global Simples (Sem Heurística)
#         # ==========================================================
#         # Extrai de forma absoluta o pixel com a maior qualidade de preensão no mapa filtrado.
#         max_idx = np.argmax(pos_out_filtered)
#         best_pixel = np.unravel_index(max_idx, pos_out_filtered.shape)
        
#         max_pixel = np.array(best_pixel)
#         grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]
        
#         rospy.loginfo_throttle(1.0, f"[GGCNN] Qualidade (Max Global): {grasp_quality:.3f}")
#         # ==========================================================
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         # ==========================================================
#         # CORREÇÃO GEOMÉTRICA 2: Cálculo Final de Abertura (Pinhole)
#         # ==========================================================
#         # O g_width repassa a largura em pixels para a classe.
#         g_width = width_px 
        
#         # O width_m assume a matemática perfeita baseada na lente da RealSense.
#         width_m = (width_px * point_depth) / (self.fx * 1000.0)
#         # ==========================================================

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         # Publica o TF do objeto na câmera apenas para translação (posição).
#         # Vamos deixar a rotação zerada aqui para não misturar matrizes.
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
        
#         # Rotação neutra. A mágica da orientação vai acontecer no base_link.
#         q = quaternion_from_euler(0.0, 0.0, 0.0) 
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         # Se a rede neural não calculou um ponto válido neste frame, aborta silenciosamente
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return None

#         try:
#             # =================================================================
#             # SOLUÇÃO DEFINITIVA CONTRA O PISCA-PISCA (Transformação Direta)
#             # =================================================================
#             # Em vez de ler o frame "object_detected" (que sofre de Race Condition),
#             # nós lemos a transformação estável da câmera em relação à base_link.
#             # Essa transformação NUNCA falha pois o robô publica as juntas constantemente.
#             cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.2))
            
#             # Criamos um ponto carimbado com as coordenadas locais da câmera.
#             # Lembre-se: grasping_point está em milímetros no seu script, convertemos para metros (/1000.0)
#             point_cam = tf2_geometry_msgs.PointStamped()
#             point_cam.header.frame_id = "camera_depth_optical_frame"
#             point_cam.header.stamp = rospy.Time(0)
#             point_cam.point.x = self.grasping_point[0] / 1000.0
#             point_cam.point.y = self.grasping_point[1] / 1000.0
#             point_cam.point.z = self.grasping_point[2] / 1000.0
            
#             # O ROS faz a matemática vetorial direta transpondo o ponto para a base_link
#             point_base = self.tf_buffer.transform(point_cam, target_frame, rospy.Duration(0.2))
#             x = point_base.point.x
#             y = point_base.point.y
#             z = point_base.point.z
            
#             # Extrai o Yaw da câmera para calcular a orientação ortogonal no plano do mundo
#             q_cam = [
#                 cam_transform.transform.rotation.x,
#                 cam_transform.transform.rotation.y,
#                 cam_transform.transform.rotation.z,
#                 cam_transform.transform.rotation.w
#             ]
#             euler_cam = euler_from_quaternion(q_cam)
#             camera_yaw = euler_cam[2]
            
#             # Mantém a nossa fórmula ortogonal perfeita com o offset corrigido
#             final_roll = math.pi 
#             final_pitch = 0.0
#             OFFSET_YAW = 1.5708    
#             final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
#             quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)
            
#             # Monta a mensagem estável para o Unity
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             # Publica sem NENHUMA perda de pacotes
#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
#             return cam_transform
            
#         except Exception as e:
#             # Caso ocorra um erro de inicialização de nós do ROS no primeiro ciclo, loga suavemente
#             rospy.logwarn_throttle(2.0, f"[GHOST GRIPPER] Aguardando sincronia de TF: {e}")
#             return None
            
#     def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
        
#         # Utiliza o quaternion recebido diretamente
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass








# #!/usr/bin/env python3

# # esta versão do script foi criada para corrigir problemas de determinação da largura do objeto detectado pelo GGCNN. A versão anterior apresentava inconsistências na medição da largura, o que poderia levar a falhas na execução da preensão pelo robô.
# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math
# from skimage.feature import peak_local_max

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         # Tratamento de buracos/NaNs na imagem de profundidade
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # ========================================================
#         # [ESTRATÉGIA DE PERCEPÇÃO] Filtro Gaussiano Espacial
#         # Mitiga o aliasing do VoxelGrid para estabilizar o vetor normal
#         # e forçar o GGCNN a encontrar ângulos ortogonais nas faces.
#         # ========================================================
#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)
#         # ========================================================

#         # Normalização e conversão para o formato da rede
#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG E SEGMENTAÇÃO)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
                        
#                         # ========================================================
#                         # SEGMENTAÇÃO DINÂMICA POR BOUNDING BOX ORIENTADO
#                         # ========================================================
#                         real_u = u_crop + offset_x
#                         real_v = v_crop + offset_y
                        
#                         # --- SOLUÇÃO ANTI-PISCAR 1: Leitura Robusta em Janela ---
#                         # Em vez de confiar num único pixel que pode ser um "buraco" (NaN),
#                         # varremos uma área 10x10 ao redor do clique e pegamos a profundidade média.
#                         min_u = max(0, real_u - 5)
#                         max_u = min(width_res, real_u + 5)
#                         min_v = max(0, real_v - 5)
#                         max_v = min(height_res, real_v + 5)
                        
#                         window = depth_copy_for_point_depth[min_v:max_v, min_u:max_u]
#                         valid_depths = window[~np.isnan(window) & (window > 0)]
                        
#                         if len(valid_depths) > 0:
#                             click_depth = np.median(valid_depths)
#                         else:
#                             click_depth = 0.0

#                         # Se o clique for muito ruidoso, usa o círculo como segurança
#                         if click_depth <= 0:
#                             cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                         else:
#                             # ========================================================
#                             # [AJUSTE] TOLERÂNCIA ASSIMÉTRICA PARA OBJETOS BAIXOS
#                             # ========================================================
#                             # Valores em milímetros. Valores menores = distância mais próxima da câmera.
#                             TOLERANCE_UP = 30.0   # Permite capturar variações/ruídos no topo da peça
#                             TOLERANCE_DOWN = 1.0 # Limite SUPER RESTRITO para baixo, evitando "vazar" para a mesa
                            
#                             # Isola a área de recorte em milímetros
#                             crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
                            
#                             # Cria a imagem binária filtrando com tolerância assimétrica
#                             binary_mask = cv2.inRange(crop_depth_mm, click_depth - TOLERANCE_UP, click_depth + TOLERANCE_DOWN)
                            
#                             # Filtro Morfológico (Opening) para remover "sujeiras" e pixels isolados da mesa
#                             kernel = np.ones((3,3), np.uint8)
#                             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
#                             # ========================================================

#                             # Encontra os Contornos da peça isolada
#                             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
#                         # # Se o clique for muito ruidoso, usa o círculo como segurança
#                         # if click_depth <= 0:
#                         #     cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                         # else:
#                         #     # Tolerância de 40mm para separar a peça da mesa
#                         #     TOLERANCE = 40.0
                            
#                         #     # Isola a área de recorte em milímetros
#                         #     crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
                            
#                         #     # Cria a imagem binária filtrando pela altura da peça
#                         #     binary_mask = cv2.inRange(crop_depth_mm, click_depth - TOLERANCE, click_depth + TOLERANCE)
                            
#                         #     # Encontra os Contornos da peça isolada
#                         #     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
#                             target_contour = None
#                             min_dist = float('inf')
                            
#                             # Acha qual contorno pertence ao clique do operador
#                             for cnt in contours:
#                                 dist = cv2.pointPolygonTest(cnt, (u_crop, v_crop), True)
#                                 if dist >= 0:
#                                     target_contour = cnt
#                                     break
#                                 elif abs(dist) < min_dist:
#                                     min_dist = abs(dist)
#                                     target_contour = cnt 
                            
#                             if target_contour is not None and len(target_contour) >= 3:
#                                 # Cria o Bounding Box Orientado exato ao redor da peça
#                                 rect = cv2.minAreaRect(target_contour)
#                                 box = cv2.boxPoints(rect)
#                                 box = np.int32(box) # np.int32 evita bugs em novas versões do Numpy
                                
#                                 # Pinta o Bounding Box de branco na máscara
#                                 cv2.drawContours(mask_2d, [box], 0, 1.0, -1)
                                
#                                 # Desenha o Bounding Box AZUL na visão de Debug
#                                 box_full = box + np.array([offset_x, offset_y])
#                                 cv2.drawContours(debug_view, [box_full], 0, (255, 0, 0), 2)
#                                 rospy.loginfo_throttle(1.0, "[GGCNN] Bounding Box Dinamico aplicado!")
#                             else:
#                                 # Se o TF calcular o ponto ligeiramente fora do crop devido a vibração do robô,
#                                 # não abortamos! Forçamos uma máscara esférica no centro da imagem (150, 150)
#                                 # pois sabemos que o robô está sobre o objeto.
#                                 cv2.circle(mask_2d, (150, 150), 95, 1.0, -1)
#                                 pos_out_filtered = pos_out_filtered * mask_2d
#                                 rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: Forçando máscara no centro.")

#                         # Multiplica o mapa de calor pela nova máscara matemática
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         # ========================================================

#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         # ==========================================================
#         # [ESTRATÉGIA DE DECISÃO] Filtro Heurístico Baseado em Física
#         # ==========================================================
#         # Extrai os top 5 picos locais
#         local_peaks = peak_local_max(pos_out_filtered, min_distance=15, num_peaks=5, threshold_abs=0.1)
        
#         if len(local_peaks) == 0:
#             rospy.logwarn_throttle(1.0, "[GGCNN] Nenhum ponto válido encontrado na imagem.")
#             return

#         best_score = -1000.0
#         best_pixel = local_peaks[0]
#         best_quality_raw = 0.0

#         # Abertura máxima da garra Robotiq 140 é 0.14m. Usamos 0.135m como margem de segurança.
#         MAX_GRIPPER_WIDTH = 0.150 
#         crop_size_width_f = float(self.crop_size)

#         for peak in local_peaks:
#             r, c = peak
#             quality = pos_out_filtered[r, c]
#             width_px_peak = abs(width_out_filtered[r, c])
            
#             # Calcula a profundidade real (Z) para este pico específico
#             reescaled_r = int(r)
#             reescaled_c = int(offset_x + c)
#             p_depth = depth_copy_for_point_depth[reescaled_r, reescaled_c]
            
#             # Se a profundidade for inválida (buraco negro do sensor), ignora o pico
#             if np.isnan(p_depth) or p_depth <= 0.01:
#                 continue
                
#             # ==========================================================
#             # CORREÇÃO GEOMÉTRICA 1: Modelo Pinhole (Óptica RealSense)
#             # ==========================================================
#             # Converte a largura em pixels para metros utilizando a distância focal (fx) exata
#             width_m_peak = (width_px_peak * p_depth) / (self.fx * 1000.0)
#             # ==========================================================
            
#             # 1. FILTRO FÍSICO: Corta preensões impossíveis (Ex: Caixa de Biscoito)
#             if width_m_peak > MAX_GRIPPER_WIDTH:
#                 rospy.loginfo_throttle(1.0, f"[Filtro] Pico descartado: Exige abertura de {width_m_peak:.3f}m (> {MAX_GRIPPER_WIDTH}m)")
#                 continue 
                
#             # 2. FILTRO DE CENTRALIDADE (ÓPTICA)
#             # Penaliza levemente preensões que estão nas bordas distorcidas da imagem.
#             # O centro do crop é (150, 150). A distância máxima possível é ~212 pixels.
#             dist_to_center = np.sqrt((r - 150)**2 + (c - 150)**2)
#             center_penalty = (dist_to_center / 150.0) * 0.05  # Penalidade suave máxima de 0.15
            
#             # O Score Final confia mais no GGCNN, mas usa a lente como desempate
#             heuristic_score = quality - center_penalty
            
#             if heuristic_score > best_score:
#                 best_score = heuristic_score
#                 best_pixel = peak
#                 best_quality_raw = quality

#         # Se todos os 5 picos forem maiores que 14cm, o sistema recusa a peça
#         if best_score == -1000.0:
#             rospy.logwarn_throttle(1.0, "[HEURISTICA] Objeto grande demais! Todos os picos excedem a abertura máxima da garra.")
#             return

#         max_pixel = np.array(best_pixel)
#         grasp_quality = best_quality_raw
        
#         rospy.loginfo_throttle(1.0, f"[HEURISTICA] Qualidade Base: {grasp_quality:.3f} | Score Final: {best_score:.3f}")
#         # ==========================================================
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         # ==========================================================
#         # CORREÇÃO GEOMÉTRICA 2: Cálculo Final de Abertura
#         # ==========================================================
#         # O g_width original assumia uma altura de mesa fixa (ROBOT_Z + 0.24), inútil no Eye-in-Hand.
#         # Agora ele apenas repassa a largura em pixels para manter a compatibilidade da classe.
#         g_width = width_px 
        
#         # O width_m assume a matemática perfeita baseada na lente da RealSense.
#         width_m = (width_px * point_depth) / (self.fx * 1000.0)
#         # ==========================================================

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         # Publica o TF do objeto na câmera apenas para translação (posição).
#         # Vamos deixar a rotação zerada aqui para não misturar matrizes.
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
        
#         # Rotação neutra. A mágica da orientação vai acontecer no base_link.
#         q = quaternion_from_euler(0.0, 0.0, 0.0) 
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         # Se a rede neural não calculou um ponto válido neste frame, aborta silenciosamente
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return None

#         try:
#             # =================================================================
#             # SOLUÇÃO DEFINITIVA CONTRA O PISCA-PISCA (Transformação Direta)
#             # =================================================================
#             # Em vez de ler o frame "object_detected" (que sofre de Race Condition),
#             # nós lemos a transformação estável da câmera em relação à base_link.
#             # Essa transformação NUNCA falha pois o robô publica as juntas constantemente.
#             cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.2))
            
#             # Criamos um ponto carimbado com as coordenadas locais da câmera.
#             # Lembre-se: grasping_point está em milímetros no seu script, convertemos para metros (/1000.0)
#             point_cam = tf2_geometry_msgs.PointStamped()
#             point_cam.header.frame_id = "camera_depth_optical_frame"
#             point_cam.header.stamp = rospy.Time(0)
#             point_cam.point.x = self.grasping_point[0] / 1000.0
#             point_cam.point.y = self.grasping_point[1] / 1000.0
#             point_cam.point.z = self.grasping_point[2] / 1000.0
            
#             # O ROS faz a matemática vetorial direta transpondo o ponto para a base_link
#             point_base = self.tf_buffer.transform(point_cam, target_frame, rospy.Duration(0.2))
#             x = point_base.point.x
#             y = point_base.point.y
#             z = point_base.point.z
            
#             # Extrai o Yaw da câmera para calcular a orientação ortogonal no plano do mundo
#             q_cam = [
#                 cam_transform.transform.rotation.x,
#                 cam_transform.transform.rotation.y,
#                 cam_transform.transform.rotation.z,
#                 cam_transform.transform.rotation.w
#             ]
#             euler_cam = euler_from_quaternion(q_cam)
#             camera_yaw = euler_cam[2]
            
#             # Mantém a nossa fórmula ortogonal perfeita com o offset corrigido
#             final_roll = math.pi 
#             final_pitch = 0.0
#             OFFSET_YAW = 1.5708    
#             final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
#             quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)
            
#             # Monta a mensagem estável para o Unity
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             # Publica sem NENHUMA perda de pacotes
#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
#             return cam_transform
            
#         except Exception as e:
#             # Caso ocorra um erro de inicialização de nós do ROS no primeiro ciclo, loga suavemente
#             rospy.logwarn_throttle(2.0, f"[GHOST GRIPPER] Aguardando sincronia de TF: {e}")
#             return None
#     def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
        
#         # Utiliza o quaternion recebido diretamente
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass












# este script estava aparesentando problemas com a determinação da largura do objeto, então foi necessário criar uma nova versão.

# #! /usr/bin/env python3

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math
# from skimage.feature import peak_local_max

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
#         self.measured_width_px = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         # Tratamento de buracos/NaNs na imagem de profundidade
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # ========================================================
#         # [ESTRATÉGIA DE PERCEPÇÃO] Filtro Gaussiano Espacial
#         # Mitiga o aliasing do VoxelGrid para estabilizar o vetor normal
#         # e forçar o GGCNN a encontrar ângulos ortogonais nas faces.
#         # ========================================================
#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)
#         # ========================================================

#         # Normalização e conversão para o formato da rede
#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG E SEGMENTAÇÃO)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
                        
#                         # ========================================================
#                         # SEGMENTAÇÃO DINÂMICA POR BOUNDING BOX ORIENTADO
#                         # ========================================================
#                         real_u = u_crop + offset_x
#                         real_v = v_crop + offset_y
                        
#                         # --- SOLUÇÃO ANTI-PISCAR 1: Leitura Robusta em Janela ---
#                         # Em vez de confiar num único pixel que pode ser um "buraco" (NaN),
#                         # varremos uma área 10x10 ao redor do clique e pegamos a profundidade média.
#                         min_u = max(0, real_u - 5)
#                         max_u = min(width_res, real_u + 5)
#                         min_v = max(0, real_v - 5)
#                         max_v = min(height_res, real_v + 5)
                        
#                         window = depth_copy_for_point_depth[min_v:max_v, min_u:max_u]
#                         valid_depths = window[~np.isnan(window) & (window > 0)]
                        
#                         if len(valid_depths) > 0:
#                             click_depth = np.median(valid_depths)
#                         else:
#                             click_depth = 0.0

#                         # Se o clique for muito ruidoso, usa o círculo como segurança
#                         if click_depth <= 0:
#                             cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                         else:
#                             # ========================================================
#                             # [AJUSTE] TOLERÂNCIA ASSIMÉTRICA PARA OBJETOS BAIXOS
#                             # ========================================================
#                             # Valores em milímetros. Valores menores = distância mais próxima da câmera.
#                             TOLERANCE_UP = 30.0   # Permite capturar variações/ruídos no topo da peça
#                             TOLERANCE_DOWN = 1.0 # Limite SUPER RESTRITO para baixo, evitando "vazar" para a mesa
                            
#                             # Isola a área de recorte em milímetros
#                             crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
                            
#                             # Cria a imagem binária filtrando com tolerância assimétrica
#                             binary_mask = cv2.inRange(crop_depth_mm, click_depth - TOLERANCE_UP, click_depth + TOLERANCE_DOWN)
                            
#                             # Filtro Morfológico (Opening) para remover "sujeiras" e pixels isolados da mesa
#                             kernel = np.ones((3,3), np.uint8)
#                             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
#                             # ========================================================

#                             # Encontra os Contornos da peça isolada
#                             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
#                         # # Se o clique for muito ruidoso, usa o círculo como segurança
#                         # if click_depth <= 0:
#                         #     cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                         # else:
#                         #     # Tolerância de 40mm para separar a peça da mesa
#                         #     TOLERANCE = 40.0
                            
#                         #     # Isola a área de recorte em milímetros
#                         #     crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
                            
#                         #     # Cria a imagem binária filtrando pela altura da peça
#                         #     binary_mask = cv2.inRange(crop_depth_mm, click_depth - TOLERANCE, click_depth + TOLERANCE)
                            
#                         #     # Encontra os Contornos da peça isolada
#                         #     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
#                             target_contour = None
#                             min_dist = float('inf')
                            
#                             # Acha qual contorno pertence ao clique do operador
#                             for cnt in contours:
#                                 dist = cv2.pointPolygonTest(cnt, (u_crop, v_crop), True)
#                                 if dist >= 0:
#                                     target_contour = cnt
#                                     break
#                                 elif abs(dist) < min_dist:
#                                     min_dist = abs(dist)
#                                     target_contour = cnt 
                            
#                             if target_contour is not None and len(target_contour) >= 3:
#                                 # Cria o Bounding Box Orientado exato ao redor da peça
#                                 rect = cv2.minAreaRect(target_contour)
#                                 box = cv2.boxPoints(rect)
#                                 box = np.int32(box) # np.int32 evita bugs em novas versões do Numpy
                                
#                                 # Pinta o Bounding Box de branco na máscara
#                                 cv2.drawContours(mask_2d, [box], 0, 1.0, -1)
                                
#                                 # Desenha o Bounding Box AZUL na visão de Debug
#                                 box_full = box + np.array([offset_x, offset_y])
#                                 cv2.drawContours(debug_view, [box_full], 0, (255, 0, 0), 2)
#                                 rospy.loginfo_throttle(1.0, "[GGCNN] Bounding Box Dinamico aplicado!")
#                             else:
#                                 # Se o TF calcular o ponto ligeiramente fora do crop devido a vibração do robô,
#                                 # não abortamos! Forçamos uma máscara esférica no centro da imagem (150, 150)
#                                 # pois sabemos que o robô está sobre o objeto.
#                                 cv2.circle(mask_2d, (150, 150), 95, 1.0, -1)
#                                 pos_out_filtered = pos_out_filtered * mask_2d
#                                 rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: Forçando máscara no centro.")

#                         # Multiplica o mapa de calor pela nova máscara matemática
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         # ========================================================

#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         # ==========================================================
#         # [ESTRATÉGIA DE DECISÃO] Filtro Heurístico Baseado em Física
#         # ==========================================================
#         # Extrai os top 5 picos locais
#         local_peaks = peak_local_max(pos_out_filtered, min_distance=15, num_peaks=5, threshold_abs=0.1)
        
#         if len(local_peaks) == 0:
#             rospy.logwarn_throttle(1.0, "[GGCNN] Nenhum ponto válido encontrado na imagem.")
#             return

#         best_score = -1000.0
#         best_pixel = local_peaks[0]
#         best_quality_raw = 0.0

#         # Abertura máxima da garra Robotiq 140 é 0.14m. Usamos 0.135m como margem de segurança.
#         MAX_GRIPPER_WIDTH = 0.150 
#         crop_size_width_f = float(self.crop_size)

#         for peak in local_peaks:
#             r, c = peak
#             quality = pos_out_filtered[r, c]
#             width_px_peak = abs(width_out_filtered[r, c])
            
#             # Calcula a profundidade real (Z) para este pico específico
#             reescaled_r = int(r)
#             reescaled_c = int(offset_x + c)
#             p_depth = depth_copy_for_point_depth[reescaled_r, reescaled_c]
            
#             # Se a profundidade for inválida (buraco negro do sensor), ignora o pico
#             if np.isnan(p_depth) or p_depth <= 0.01:
#                 continue
                
#             # Calcula o tamanho real da peça em metros baseada no FOV e Profundidade
#             width_m_peak = (width_px_peak / crop_size_width_f) * 2.0 * p_depth * np.tan(self.FOV * crop_size_width_f / height_res / 2.0 / 180.0 * np.pi) / 1000.0
            
#             # 1. FILTRO FÍSICO: Corta preensões impossíveis (Ex: Caixa de Biscoito)
#             if width_m_peak > MAX_GRIPPER_WIDTH:
#                 rospy.loginfo_throttle(1.0, f"[Filtro] Pico descartado: Exige abertura de {width_m_peak:.3f}m (> {MAX_GRIPPER_WIDTH}m)")
#                 continue 
                
#             # 2. FILTRO DE CENTRALIDADE (ÓPTICA)
#             # Penaliza levemente preensões que estão nas bordas distorcidas da imagem.
#             # O centro do crop é (150, 150). A distância máxima possível é ~212 pixels.
#             dist_to_center = np.sqrt((r - 150)**2 + (c - 150)**2)
#             center_penalty = (dist_to_center / 150.0) * 0.05  # Penalidade suave máxima de 0.15
            
#             # O Score Final confia mais no GGCNN, mas usa a lente como desempate
#             heuristic_score = quality - center_penalty
            
#             if heuristic_score > best_score:
#                 best_score = heuristic_score
#                 best_pixel = peak
#                 best_quality_raw = quality

#         # Se todos os 5 picos forem maiores que 14cm, o sistema recusa a peça
#         if best_score == -1000.0:
#             rospy.logwarn_throttle(1.0, "[HEURISTICA] Objeto grande demais! Todos os picos excedem a abertura máxima da garra.")
#             return

#         max_pixel = np.array(best_pixel)
#         grasp_quality = best_quality_raw
        
#         rospy.loginfo_throttle(1.0, f"[HEURISTICA] Qualidade Base: {grasp_quality:.3f} | Score Final: {best_score:.3f}")
#         # ==========================================================
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) 
#         crop_size_width = float(self.crop_size)
#         width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 
#         width_m = abs(width_m[max_pixel[0], max_pixel[1]])

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         # Publica o TF do objeto na câmera apenas para translação (posição).
#         # Vamos deixar a rotação zerada aqui para não misturar matrizes.
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
        
#         # Rotação neutra. A mágica da orientação vai acontecer no base_link.
#         q = quaternion_from_euler(0.0, 0.0, 0.0) 
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         # Se a rede neural não calculou um ponto válido neste frame, aborta silenciosamente
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return None

#         try:
#             # =================================================================
#             # SOLUÇÃO DEFINITIVA CONTRA O PISCA-PISCA (Transformação Direta)
#             # =================================================================
#             # Em vez de ler o frame "object_detected" (que sofre de Race Condition),
#             # nós lemos a transformação estável da câmera em relação à base_link.
#             # Essa transformação NUNCA falha pois o robô publica as juntas constantemente.
#             cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.2))
            
#             # Criamos um ponto carimbado com as coordenadas locais da câmera.
#             # Lembre-se: grasping_point está em milímetros no seu script, convertemos para metros (/1000.0)
#             point_cam = tf2_geometry_msgs.PointStamped()
#             point_cam.header.frame_id = "camera_depth_optical_frame"
#             point_cam.header.stamp = rospy.Time(0)
#             point_cam.point.x = self.grasping_point[0] / 1000.0
#             point_cam.point.y = self.grasping_point[1] / 1000.0
#             point_cam.point.z = self.grasping_point[2] / 1000.0
            
#             # O ROS faz a matemática vetorial direta transpondo o ponto para a base_link
#             point_base = self.tf_buffer.transform(point_cam, target_frame, rospy.Duration(0.2))
#             x = point_base.point.x
#             y = point_base.point.y
#             z = point_base.point.z
            
#             # Extrai o Yaw da câmera para calcular a orientação ortogonal no plano do mundo
#             q_cam = [
#                 cam_transform.transform.rotation.x,
#                 cam_transform.transform.rotation.y,
#                 cam_transform.transform.rotation.z,
#                 cam_transform.transform.rotation.w
#             ]
#             euler_cam = euler_from_quaternion(q_cam)
#             camera_yaw = euler_cam[2]
            
#             # Mantém a nossa fórmula ortogonal perfeita com o offset corrigido
#             final_roll = math.pi 
#             final_pitch = 0.0
#             OFFSET_YAW = 1.5708    
#             final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
#             quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)
            
#             # Monta a mensagem estável para o Unity
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             # Publica sem NENHUMA perda de pacotes
#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
#             return cam_transform
            
#         except Exception as e:
#             # Caso ocorra um erro de inicialização de nós do ROS no primeiro ciclo, loga suavemente
#             rospy.logwarn_throttle(2.0, f"[GHOST GRIPPER] Aguardando sincronia de TF: {e}")
#             return None
#     def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
        
#         # Utiliza o quaternion recebido diretamente
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass

















# #! /usr/bin/env python3

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math
# from skimage.feature import peak_local_max

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         # Tratamento de buracos/NaNs na imagem de profundidade
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # ========================================================
#         # [ESTRATÉGIA DE PERCEPÇÃO] Filtro Gaussiano Espacial
#         # Mitiga o aliasing do VoxelGrid para estabilizar o vetor normal
#         # e forçar o GGCNN a encontrar ângulos ortogonais nas faces.
#         # ========================================================
#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)
#         # ========================================================

#         # Normalização e conversão para o formato da rede
#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG E SEGMENTAÇÃO)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
                        
#                         # ========================================================
#                         # SEGMENTAÇÃO DINÂMICA POR BOUNDING BOX ORIENTADO
#                         # ========================================================
#                         real_u = u_crop + offset_x
#                         real_v = v_crop + offset_y
                        
#                         # Garante que os índices estão dentro dos limites da câmera
#                         if 0 <= real_u < width_res and 0 <= real_v < height_res:
#                             click_depth = depth_copy_for_point_depth[real_v, real_u]
#                         else:
#                             click_depth = 0.0
                            
#                         # Se o clique for muito ruidoso/NaN, usa o círculo como segurança
#                         if np.isnan(click_depth) or click_depth <= 0:
#                             cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                             rospy.logwarn_throttle(1.0, "[GGCNN] Profundidade invalida. Usando circulo (Fallback).")
#                         else:
#                             # Tolerância de 40mm para separar a peça da mesa
#                             TOLERANCE = 40.0 
                            
#                             # Isola a área de recorte em milímetros
#                             crop_depth_mm = depth_copy_for_point_depth[offset_y:offset_y+self.crop_size, offset_x:offset_x+self.crop_size]
                            
#                             # Cria a imagem binária filtrando pela altura da peça
#                             binary_mask = cv2.inRange(crop_depth_mm, click_depth - TOLERANCE, click_depth + TOLERANCE)
                            
#                             # Encontra os Contornos da peça isolada
#                             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
#                             target_contour = None
#                             min_dist = float('inf')
                            
#                             # Acha qual contorno pertence ao clique do operador
#                             for cnt in contours:
#                                 dist = cv2.pointPolygonTest(cnt, (u_crop, v_crop), True)
#                                 if dist >= 0:
#                                     target_contour = cnt
#                                     break
#                                 elif abs(dist) < min_dist:
#                                     min_dist = abs(dist)
#                                     target_contour = cnt 
                            
#                             if target_contour is not None and len(target_contour) >= 3:
#                                 # Cria o Bounding Box Orientado exato ao redor da peça
#                                 rect = cv2.minAreaRect(target_contour)
#                                 box = cv2.boxPoints(rect)
#                                 box = np.int32(box) # np.int32 evita bugs em novas versões do Numpy
                                
#                                 # Pinta o Bounding Box de branco na máscara
#                                 cv2.drawContours(mask_2d, [box], 0, 1.0, -1)
                                
#                                 # Desenha o Bounding Box AZUL na visão de Debug
#                                 box_full = box + np.array([offset_x, offset_y])
#                                 cv2.drawContours(debug_view, [box_full], 0, (255, 0, 0), 2)
#                                 rospy.loginfo_throttle(1.0, "[GGCNN] Bounding Box Dinamico aplicado!")
#                             else:
#                                 cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                                 rospy.logwarn_throttle(1.0, "[GGCNN] Falha ao achar bordas. Usando circulo.")

#                         # Multiplica o mapa de calor pela nova máscara matemática
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         # ========================================================

#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         # ==========================================================
#         # [ESTRATÉGIA DE DECISÃO] Filtro Heurístico Baseado em Física
#         # ==========================================================
#         # Extrai os top 5 picos locais
#         local_peaks = peak_local_max(pos_out_filtered, min_distance=15, num_peaks=5, threshold_abs=0.1)
        
#         if len(local_peaks) == 0:
#             rospy.logwarn_throttle(1.0, "[GGCNN] Nenhum ponto válido encontrado na imagem.")
#             return

#         best_score = -1000.0
#         best_pixel = local_peaks[0]
#         best_quality_raw = 0.0

#         # Abertura máxima da garra Robotiq 140 é 0.14m. Usamos 0.135m como margem de segurança.
#         MAX_GRIPPER_WIDTH = 0.135 
#         crop_size_width_f = float(self.crop_size)

#         for peak in local_peaks:
#             r, c = peak
#             quality = pos_out_filtered[r, c]
#             width_px_peak = abs(width_out_filtered[r, c])
            
#             # Calcula a profundidade real (Z) para este pico específico
#             reescaled_r = int(r)
#             reescaled_c = int(offset_x + c)
#             p_depth = depth_copy_for_point_depth[reescaled_r, reescaled_c]
            
#             # Se a profundidade for inválida (buraco negro do sensor), ignora o pico
#             if np.isnan(p_depth) or p_depth <= 0.01:
#                 continue
                
#             # Calcula o tamanho real da peça em metros baseada no FOV e Profundidade
#             width_m_peak = (width_px_peak / crop_size_width_f) * 2.0 * p_depth * np.tan(self.FOV * crop_size_width_f / height_res / 2.0 / 180.0 * np.pi) / 1000.0
            
#             # 1. FILTRO FÍSICO: Corta preensões impossíveis (Ex: Caixa de Biscoito)
#             if width_m_peak > MAX_GRIPPER_WIDTH:
#                 rospy.loginfo_throttle(1.0, f"[Filtro] Pico descartado: Exige abertura de {width_m_peak:.3f}m (> {MAX_GRIPPER_WIDTH}m)")
#                 continue 
                
#             # 2. FILTRO DE CENTRALIDADE (ÓPTICA)
#             # Penaliza levemente preensões que estão nas bordas distorcidas da imagem.
#             # O centro do crop é (150, 150). A distância máxima possível é ~212 pixels.
#             dist_to_center = np.sqrt((r - 150)**2 + (c - 150)**2)
#             center_penalty = (dist_to_center / 150.0) * 0.15  # Penalidade suave máxima de 0.15
            
#             # O Score Final confia mais no GGCNN, mas usa a lente como desempate
#             heuristic_score = quality - center_penalty
            
#             if heuristic_score > best_score:
#                 best_score = heuristic_score
#                 best_pixel = peak
#                 best_quality_raw = quality

#         # Se todos os 5 picos forem maiores que 14cm, o sistema recusa a peça
#         if best_score == -1000.0:
#             rospy.logwarn_throttle(1.0, "[HEURISTICA] Objeto grande demais! Todos os picos excedem a abertura máxima da garra.")
#             return

#         max_pixel = np.array(best_pixel)
#         grasp_quality = best_quality_raw
        
#         rospy.loginfo_throttle(1.0, f"[HEURISTICA] Qualidade Base: {grasp_quality:.3f} | Score Final: {best_score:.3f}")
#         # ==========================================================
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) 
#         crop_size_width = float(self.crop_size)
#         width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 
#         width_m = abs(width_m[max_pixel[0], max_pixel[1]])

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         # Publica o TF do objeto na câmera apenas para translação (posição).
#         # Vamos deixar a rotação zerada aqui para não misturar matrizes.
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
        
#         # Rotação neutra. A mágica da orientação vai acontecer no base_link.
#         q = quaternion_from_euler(0.0, 0.0, 0.0) 
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         try:
#             # 1. Pega a posição exata do objeto já resolvida para o base_link
#             transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
#             x = transform.transform.translation.x
#             y = transform.transform.translation.y
#             z = transform.transform.translation.z   
            
#             # 2. Descobre para onde a câmera está apontando (Yaw) no mundo
#             cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.1))
#             q_cam = [
#                 cam_transform.transform.rotation.x,
#                 cam_transform.transform.rotation.y,
#                 cam_transform.transform.rotation.z,
#                 cam_transform.transform.rotation.w
#             ]
#             euler_cam = euler_from_quaternion(q_cam)
#             camera_yaw = euler_cam[2] # Extrai apenas o eixo Z (Yaw) da câmera
            
#             # =========================================================
#             # 3. A FÓRMULA ORTOGONAL (Correção Absoluta)
#             # =========================================================
#             final_roll = math.pi 
#             final_pitch = 0.0
            
#             # --- AJUSTE DE MAPEAMENTO 2D PARA 3D ---
#             # Se o heatmap está certo e o Unity está errado, o problema é o mapeamento.
#             # Em muitas câmeras RealSense olhando para baixo, o eixo angular é INVERTIDO
#             # e defasado em 90 graus em relação ao referencial global (base_link).
            
#             # Tente esta configuração primária (+ self.ang em vez de - self.ang):
#             OFFSET_YAW = 1.5708  # Adiciona 90 graus (em radianos) para cruzar o eixo
            
#             # A equação agora SOMA o ângulo da rede em vez de subtrair, 
#             # corrigindo a inversão de eixo do plano da imagem para o plano 3D.
#             final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
#             quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)

#             # =========================================================
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
#             return transform
            
#         except Exception as e:
#             # Erros de TF não imprimem no console para não dar lag, mas passam silenciosamente.
#             return None

#     def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
        
#         # Utiliza o quaternion recebido diretamente
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass


















# #! /usr/bin/env python3

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math
# from skimage.feature import peak_local_max

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         # Tratamento de buracos/NaNs na imagem de profundidade
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # ========================================================
#         # [ESTRATÉGIA DE PERCEPÇÃO] Filtro Gaussiano Espacial
#         # Mitiga o aliasing do VoxelGrid para estabilizar o vetor normal
#         # e forçar o GGCNN a encontrar ângulos ortogonais nas faces.
#         # ========================================================
#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)
#         # ========================================================

#         # Normalização e conversão para o formato da rede
#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
#                         cv2.circle(mask_2d, (u_crop, v_crop), 95, 1.0, -1)
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         rospy.loginfo_throttle(1.0, "[GGCNN] DENTRO DO CROP: Mascara aplicada!")
#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico (Imagem da câmera + ponto projetado)
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         # ==========================================================
#         # [ESTRATÉGIA DE DECISÃO] Filtro Heurístico Baseado em Física
#         # ==========================================================
#         # Extrai os top 5 picos locais
#         local_peaks = peak_local_max(pos_out_filtered, min_distance=15, num_peaks=5, threshold_abs=0.1)
        
#         if len(local_peaks) == 0:
#             rospy.logwarn_throttle(1.0, "[GGCNN] Nenhum ponto válido encontrado na imagem.")
#             return

#         best_score = -1000.0
#         best_pixel = local_peaks[0]
#         best_quality_raw = 0.0

#         # Abertura máxima da garra Robotiq 140 é 0.14m. Usamos 0.135m como margem de segurança.
#         MAX_GRIPPER_WIDTH = 0.135 
#         crop_size_width_f = float(self.crop_size)

#         for peak in local_peaks:
#             r, c = peak
#             quality = pos_out_filtered[r, c]
#             width_px_peak = abs(width_out_filtered[r, c])
            
#             # Calcula a profundidade real (Z) para este pico específico
#             reescaled_r = int(r)
#             reescaled_c = int(offset_x + c)
#             p_depth = depth_copy_for_point_depth[reescaled_r, reescaled_c]
            
#             # Se a profundidade for inválida (buraco negro do sensor), ignora o pico
#             if np.isnan(p_depth) or p_depth <= 0.01:
#                 continue
                
#             # Calcula o tamanho real da peça em metros baseada no FOV e Profundidade
#             width_m_peak = (width_px_peak / crop_size_width_f) * 2.0 * p_depth * np.tan(self.FOV * crop_size_width_f / height_res / 2.0 / 180.0 * np.pi) / 1000.0
            
#             # 1. FILTRO FÍSICO: Corta preensões impossíveis (Ex: Caixa de Biscoito)
#             if width_m_peak > MAX_GRIPPER_WIDTH:
#                 rospy.loginfo_throttle(1.0, f"[Filtro] Pico descartado: Exige abertura de {width_m_peak:.3f}m (> {MAX_GRIPPER_WIDTH}m)")
#                 continue 
                
#             # 2. FILTRO DE CENTRALIDADE (ÓPTICA)
#             # Penaliza levemente preensões que estão nas bordas distorcidas da imagem.
#             # O centro do crop é (150, 150). A distância máxima possível é ~212 pixels.
#             dist_to_center = np.sqrt((r - 150)**2 + (c - 150)**2)
#             center_penalty = (dist_to_center / 150.0) * 0.15  # Penalidade suave máxima de 0.15
            
#             # O Score Final confia mais no GGCNN, mas usa a lente como desempate
#             heuristic_score = quality - center_penalty
            
#             if heuristic_score > best_score:
#                 best_score = heuristic_score
#                 best_pixel = peak
#                 best_quality_raw = quality

#         # Se todos os 5 picos forem maiores que 14cm, o sistema recusa a peça
#         if best_score == -1000.0:
#             rospy.logwarn_throttle(1.0, "[HEURISTICA] Objeto grande demais! Todos os picos excedem a abertura máxima da garra.")
#             return

#         max_pixel = np.array(best_pixel)
#         grasp_quality = best_quality_raw
        
#         rospy.loginfo_throttle(1.0, f"[HEURISTICA] Qualidade Base: {grasp_quality:.3f} | Score Final: {best_score:.3f}")
#         # ==========================================================
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) 
#         crop_size_width = float(self.crop_size)
#         width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 
#         width_m = abs(width_m[max_pixel[0], max_pixel[1]])

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         # Publica o TF do objeto na câmera apenas para translação (posição).
#         # Vamos deixar a rotação zerada aqui para não misturar matrizes.
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
        
#         # Rotação neutra. A mágica da orientação vai acontecer no base_link.
#         q = quaternion_from_euler(0.0, 0.0, 0.0) 
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         try:
#             # 1. Pega a posição exata do objeto já resolvida para o base_link
#             transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
#             x = transform.transform.translation.x
#             y = transform.transform.translation.y
#             z = transform.transform.translation.z   
            
#             # 2. Descobre para onde a câmera está apontando (Yaw) no mundo
#             cam_transform = self.tf_buffer.lookup_transform(target_frame, "camera_depth_optical_frame", rospy.Time(0), rospy.Duration(0.1))
#             q_cam = [
#                 cam_transform.transform.rotation.x,
#                 cam_transform.transform.rotation.y,
#                 cam_transform.transform.rotation.z,
#                 cam_transform.transform.rotation.w
#             ]
#             euler_cam = euler_from_quaternion(q_cam)
#             camera_yaw = euler_cam[2] # Extrai apenas o eixo Z (Yaw) da câmera
            
#             # =========================================================
#             # 3. A FÓRMULA ORTOGONAL (Correção Absoluta)
#             # =========================================================
#             final_roll = math.pi 
#             final_pitch = 0.0
            
#             # --- AJUSTE DE MAPEAMENTO 2D PARA 3D ---
#             # Se o heatmap está certo e o Unity está errado, o problema é o mapeamento.
#             # Em muitas câmeras RealSense olhando para baixo, o eixo angular é INVERTIDO
#             # e defasado em 90 graus em relação ao referencial global (base_link).
            
#             # Tente esta configuração primária (+ self.ang em vez de - self.ang):
#             OFFSET_YAW = 1.5708  # Adiciona 90 graus (em radianos) para cruzar o eixo
            
#             # A equação agora SOMA o ângulo da rede em vez de subtrair, 
#             # corrigindo a inversão de eixo do plano da imagem para o plano 3D.
#             final_yaw = camera_yaw + self.ang + OFFSET_YAW
            
#             quat = quaternion_from_euler(final_roll, final_pitch, final_yaw)

#             # =========================================================
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, quat, 'base_link', 'object_grasp')
#             return transform
            
#         except Exception as e:
#             # Erros de TF não imprimem no console para não dar lag, mas passam silenciosamente.
#             return None

#     def publish_static_transform(self, x, y, z, quat, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
        
#         # Utiliza o quaternion recebido diretamente
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass


















# #! /usr/bin/env python3

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math
# from skimage.feature import peak_local_max

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         # Tratamento de buracos/NaNs na imagem de profundidade
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # ========================================================
#         # [ESTRATÉGIA DE PERCEPÇÃO] Filtro Gaussiano Espacial
#         # Mitiga o aliasing do VoxelGrid para estabilizar o vetor normal
#         # e forçar o GGCNN a encontrar ângulos ortogonais nas faces.
#         # ========================================================
#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)
#         # ========================================================

#         # Normalização e conversão para o formato da rede
#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
#                         cv2.circle(mask_2d, (u_crop, v_crop), 45, 1.0, -1)
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         rospy.loginfo_throttle(1.0, "[GGCNN] DENTRO DO CROP: Mascara aplicada!")
#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico (Imagem da câmera + ponto projetado)
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         # ==========================================================
#         # [ESTRATÉGIA DE DECISÃO] Filtro Heurístico Baseado em Física
#         # ==========================================================
#         # Extrai os top 5 picos locais
#         local_peaks = peak_local_max(pos_out_filtered, min_distance=15, num_peaks=5, threshold_abs=0.1)
        
#         if len(local_peaks) == 0:
#             rospy.logwarn_throttle(1.0, "[GGCNN] Nenhum ponto válido encontrado na imagem.")
#             return

#         best_score = -1000.0
#         best_pixel = local_peaks[0]
#         best_quality_raw = 0.0

#         # Abertura máxima da garra Robotiq 140 é 0.14m. Usamos 0.135m como margem de segurança.
#         MAX_GRIPPER_WIDTH = 0.135 
#         crop_size_width_f = float(self.crop_size)

#         for peak in local_peaks:
#             r, c = peak
#             quality = pos_out_filtered[r, c]
#             width_px_peak = abs(width_out_filtered[r, c])
            
#             # Calcula a profundidade real (Z) para este pico específico
#             reescaled_r = int(r)
#             reescaled_c = int(offset_x + c)
#             p_depth = depth_copy_for_point_depth[reescaled_r, reescaled_c]
            
#             # Se a profundidade for inválida (buraco negro do sensor), ignora o pico
#             if np.isnan(p_depth) or p_depth <= 0.01:
#                 continue
                
#             # Calcula o tamanho real da peça em metros baseada no FOV e Profundidade
#             width_m_peak = (width_px_peak / crop_size_width_f) * 2.0 * p_depth * np.tan(self.FOV * crop_size_width_f / height_res / 2.0 / 180.0 * np.pi) / 1000.0
            
#             # 1. FILTRO FÍSICO: Corta preensões impossíveis (Ex: Caixa de Biscoito)
#             if width_m_peak > MAX_GRIPPER_WIDTH:
#                 rospy.loginfo_throttle(1.0, f"[Filtro] Pico descartado: Exige abertura de {width_m_peak:.3f}m (> {MAX_GRIPPER_WIDTH}m)")
#                 continue 
                
#             # 2. FILTRO DE CENTRALIDADE (ÓPTICA)
#             # Penaliza levemente preensões que estão nas bordas distorcidas da imagem.
#             # O centro do crop é (150, 150). A distância máxima possível é ~212 pixels.
#             dist_to_center = np.sqrt((r - 150)**2 + (c - 150)**2)
#             center_penalty = (dist_to_center / 150.0) * 0.15  # Penalidade suave máxima de 0.15
            
#             # O Score Final confia mais no GGCNN, mas usa a lente como desempate
#             heuristic_score = quality - center_penalty
            
#             if heuristic_score > best_score:
#                 best_score = heuristic_score
#                 best_pixel = peak
#                 best_quality_raw = quality

#         # Se todos os 5 picos forem maiores que 14cm, o sistema recusa a peça
#         if best_score == -1000.0:
#             rospy.logwarn_throttle(1.0, "[HEURISTICA] Objeto grande demais! Todos os picos excedem a abertura máxima da garra.")
#             return

#         max_pixel = np.array(best_pixel)
#         grasp_quality = best_quality_raw
        
#         rospy.loginfo_throttle(1.0, f"[HEURISTICA] Qualidade Base: {grasp_quality:.3f} | Score Final: {best_score:.3f}")
#         # ==========================================================
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) 
#         crop_size_width = float(self.crop_size)
#         width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 
#         width_m = abs(width_m[max_pixel[0], max_pixel[1]])

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
#         q = quaternion_from_euler(3.14, 0, -1*cmd_msg.data[3])
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         try:
#             transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
            
#             x = transform.transform.translation.x
#             y = transform.transform.translation.y
#             z = transform.transform.translation.z   
            
#             roll = 3.140
#             pitch = 0.0
#             yaw = -1 * self.ang  
#             quat = quaternion_from_euler(roll, pitch, yaw)
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, roll, pitch, yaw, 'base_link', 'object_grasp')
#             return transform
#         except:
#             return None

#     def publish_static_transform(self, x, y, z, roll, pitch, yaw, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
#         quat = quaternion_from_euler(roll, pitch, yaw)
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass













# #! /usr/bin/env python3

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         # Tratamento de buracos/NaNs na imagem de profundidade
#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         # ========================================================
#         # [ESTRATÉGIA DE PERCEPÇÃO] Filtro Gaussiano Espacial
#         # Mitiga o aliasing do VoxelGrid para estabilizar o vetor normal
#         # e forçar o GGCNN a encontrar ângulos ortogonais nas faces.
#         # ========================================================
#         depth_crop = cv2.GaussianBlur(depth_crop, (5, 5), 0)
#         # ========================================================

#         # Normalização e conversão para o formato da rede
#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
#                         cv2.circle(mask_2d, (u_crop, v_crop), 45, 1.0, -1)
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         rospy.loginfo_throttle(1.0, "[GGCNN] DENTRO DO CROP: Mascara aplicada!")
#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico (Imagem da câmera + ponto projetado)
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         max_pixel = np.array(np.unravel_index(np.argmax(pos_out_filtered), pos_out_filtered.shape)) 
#         grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]   
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) 
#         crop_size_width = float(self.crop_size)
#         width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 
#         width_m = abs(width_m[max_pixel[0], max_pixel[1]])

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
#         q = quaternion_from_euler(3.14, 0, -1*cmd_msg.data[3])
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         try:
#             transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
            
#             x = transform.transform.translation.x
#             y = transform.transform.translation.y
#             z = transform.transform.translation.z   
            
#             roll = 3.140
#             pitch = 0.0
#             yaw = -1 * self.ang  
#             quat = quaternion_from_euler(roll, pitch, yaw)
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, roll, pitch, yaw, 'base_link', 'object_grasp')
#             return transform
#         except:
#             return None

#     def publish_static_transform(self, x, y, z, roll, pitch, yaw, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
#         quat = quaternion_from_euler(roll, pitch, yaw)
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass















# #! /usr/bin/env python3

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         # ========================================================
#         # TRAVA ANTI-COLAPSO: Filtra pixels que vazaram do crop 300x300
#         # ========================================================
#         # Verifica quais índices são maiores que 0 E menores que o limite da imagem
#         valid_indices = (rr >= 0) & (rr < qual_img.shape[0]) & (cc >= 0) & (cc < qual_img.shape[1])
        
#         # Pinta com 255 APENAS os pixels que passaram no teste
#         qual_img[rr[valid_indices], cc[valid_indices]] = 255
#         # ========================================================
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
#                         cv2.circle(mask_2d, (u_crop, v_crop), 45, 1.0, -1)
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         rospy.loginfo_throttle(1.0, "[GGCNN] DENTRO DO CROP: Mascara aplicada!")
#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico (Imagem da câmera + ponto projetado)
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         max_pixel = np.array(np.unravel_index(np.argmax(pos_out_filtered), pos_out_filtered.shape)) 
#         grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]   
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) 
#         crop_size_width = float(self.crop_size)
#         width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 
#         width_m = abs(width_m[max_pixel[0], max_pixel[1]])

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang 
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered  

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
#         q = quaternion_from_euler(3.14, 0, -1*cmd_msg.data[3])
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         try:
#             transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
            
#             x = transform.transform.translation.x
#             y = transform.transform.translation.y
#             z = transform.transform.translation.z   
            
#             roll = 3.140
#             pitch = 0.0
#             yaw = -1 * self.ang  
#             quat = quaternion_from_euler(roll, pitch, yaw)
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, roll, pitch, yaw, 'base_link', 'object_grasp')
#             return transform
#         except:
#             return None

#     def publish_static_transform(self, x, y, z, roll, pitch, yaw, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
#         quat = quaternion_from_euler(roll, pitch, yaw)
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass
















# #! /usr/bin/env python3

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 
#         # ==================================================================
#         # NOVO: Publicador de Visão Ativa (Comanda o robô a se mover)
#         # ==================================================================
#         self.hover_pub = rospy.Publisher('unity/target_pose', PoseStamped, queue_size=1)
#         # ==================================================================        

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#         # ==================================================================
#         # AUTONOMIA COMPARTILHADA: AÇÃO DE VISÃO ATIVA (HOVER POSE)
#         # ==================================================================
#         hover_pose = PoseStamped()
#         hover_pose.header.stamp = rospy.Time.now()
#         hover_pose.header.frame_id = "base_link"
        
#         # Posição: X e Y do clique, mas com Z fixado em 40cm (Segurança)
#         hover_pose.pose.position.x = correct_x
#         hover_pose.pose.position.y = correct_y
#         hover_pose.pose.position.z = 0.40 
        
#         # Orientação: Câmera apontando estritamente para baixo 
#         # (Utilizando o padrão do seu TEST_MODE: Roll=0, Pitch=1.57, Yaw=0)
#         quat = quaternion_from_euler(0.0, 1.5708, 0.0)
#         hover_pose.pose.orientation.x = quat[0]
#         hover_pose.pose.orientation.y = quat[1]
#         hover_pose.pose.orientation.z = quat[2]
#         hover_pose.pose.orientation.w = quat[3]
        
#         # Comanda o KDL Solver a mover o braço
#         self.hover_pub.publish(hover_pose)
#         rospy.loginfo("[VISÃO ATIVA] Robô movendo-se para centralizar o alvo no FOV!")
#         # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         qual_img[rr, cc] = 255
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
#                         cv2.circle(mask_2d, (u_crop, v_crop), 45, 1.0, -1)
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         rospy.loginfo_throttle(1.0, "[GGCNN] DENTRO DO CROP: Mascara aplicada!")
#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico (Imagem da câmera + ponto projetado)
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         max_pixel = np.array(np.unravel_index(np.argmax(pos_out_filtered), pos_out_filtered.shape)) 
#         grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]   
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) 
#         crop_size_width = float(self.crop_size)
#         width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 
#         width_m = abs(width_m[max_pixel[0], max_pixel[1]])

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered  

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
#         q = quaternion_from_euler(3.14, 0, -1*cmd_msg.data[3])
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         try:
#             transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
            
#             x = transform.transform.translation.x
#             y = transform.transform.translation.y
#             z = transform.transform.translation.z   
            
#             roll = 3.140
#             pitch = 0.0
#             yaw = -1 * self.ang  
#             quat = quaternion_from_euler(roll, pitch, yaw)
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, roll, pitch, yaw, 'base_link', 'object_grasp')
#             return transform
#         except:
#             return None

#     def publish_static_transform(self, x, y, z, roll, pitch, yaw, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
#         quat = quaternion_from_euler(roll, pitch, yaw)
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass
















# #! /usr/bin/env python3

# import time
# import numpy as np
# import argparse
# from skimage.draw import circle_perimeter

# import torch
# import cv2
# import tf2_ros
# import tf2_geometry_msgs

# import rospy
# import rospkg
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image, CameraInfo
# from std_msgs.msg import Float32MultiArray, Float32
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from geometry_msgs.msg import TransformStamped, PoseStamped, PointStamped, Point
# import math

# from models.ggcnn import GGCNN 

# class ggcnn_grasping(object):
#     def __init__(self, args):
#         rospy.init_node('ggcnn_detection')

#         self.args = args
#         self.bridge = CvBridge()
#         self.latest_depth_message = None
#         self.color_img = None
        
#         rospack = rospkg.RosPack()
#         Home = rospack.get_path('ggcnn_pkg')
#         MODEL_FILE = Home + '/scripts/ggcnn_grasping/models_trined/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
#         self.model = GGCNN()
#         self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
#         self.model.eval()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()

#         self.crop_size = rospy.get_param("/GGCNN/crop_size", 300)
#         self.FOV = rospy.get_param("/GGCNN/FOV", 60)
#         self.camera_topic_info = rospy.get_param("/GGCNN/camera_topic_info", "/camera/depth/camera_info")
        
#         if self.args.real:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense", "/camera/depth/image_raw")
#         else:
#             self.camera_topic = rospy.get_param("/GGCNN/camera_topic", "/camera/depth/image_raw")

#         self.grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
#         self.depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
#         self.width_pub = rospy.Publisher('ggcnn/img/width', Image, queue_size=1)
#         self.depth_pub_shot = rospy.Publisher('ggcnn/img/depth_shot', Image, queue_size=1)
#         self.ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
#         self.cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)
#         self.cmd_pub_grasp = rospy.Publisher('ggcnn/out/command_grasp', Float32MultiArray, queue_size=1) 
#         self.unity_pose_pub = rospy.Publisher('/ggcnn/unity_target_pose', PoseStamped, queue_size=1)  
#         self.unity_width_pub = rospy.Publisher('/ggcnn/unity_gripper_width', Float32, queue_size=1) 

#         self.grasping_point = []
#         self.depth_image_shot = None
        
#         # VARIÁVEL DE INTENÇÃO DO VR
#         self.unity_target_base_link = None
        
#         camera_info_msg = rospy.wait_for_message(self.camera_topic_info, CameraInfo)
#         K = camera_info_msg.K
#         self.fx = K[0]
#         self.cx = K[2]
#         self.fy = K[4]
#         self.cy = K[5]

#         # Os Subscribers do ROS
#         rospy.Subscriber(self.camera_topic, Image, self.get_depth_callback, queue_size=10)
#         rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
#         rospy.Subscriber('/ggcnn/target_intention_point', Point, self.intention_callback, queue_size=1)

#     # ==================================================================
#     # NOVA FUNÇÃO DE ROTAÇÃO DOS EIXOS
#     # ==================================================================
#     def intention_callback(self, msg):
#         # Remapeamento matemático exato para o ROS (base_link):
#         correct_x = msg.z   # O eixo X (Frente) recebe a profundidade
#         correct_y = -msg.y  # O eixo Y (Lateral) é invertido
#         correct_z = msg.x   # O eixo Z (Cima) recebe a altura
        
#         self.unity_target_base_link = [correct_x, correct_y, correct_z]
        
#         rospy.loginfo_throttle(2.0, f"[UNITY CORRIGIDO] X={correct_x:.3f}, Y={correct_y:.3f}, Z={correct_z:.3f}")

#     # ==================================================================
#     def get_depth_callback(self, depth_message):
#         self.latest_depth_message = depth_message

#     def image_callback(self, color_msg):
#         self.color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

#     def draw_grasp(self, img, x, y, theta, width, height=20, color=(0,255,0), thickness=2):
#         dx = width / 2
#         dy = height / 2
#         rect = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]])
#         R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
#         rect = rect @ R.T
#         rect[:, 0] += x
#         rect[:, 1] += y
#         rect = rect.astype(np.int32)
#         cv2.polylines(img, [rect], isClosed=True, color=color, thickness=thickness)
#         return img

#     def _normalize_and_colorize_map(self, map_array, min_val=0, max_val=1):
#         if np.max(map_array) > np.min(map_array):
#             normalized_map = (map_array - np.min(map_array)) / (np.max(map_array) - np.min(map_array))
#         else:
#             normalized_map = np.zeros_like(map_array, dtype=np.float32)
#         normalized_map = np.clip(normalized_map * 255, 0, 255).astype(np.uint8)
#         normalized_map = np.ascontiguousarray(normalized_map)
#         colorized_map = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
#         return colorized_map

#     def _generate_visual_maps(self, pos_out, ang_out, width_out, qual_out):
#         pos_img = self._normalize_and_colorize_map(pos_out)
#         ang_img = self._normalize_and_colorize_map(ang_out, min_val=-np.pi/2, max_val=np.pi/2)
#         width_img = self._normalize_and_colorize_map(width_out)
#         qual_img = self._normalize_and_colorize_map(qual_out)
        
#         qual_img = self.draw_grasp(qual_img, self.best_x, self.best_y, -1*self.ang, self.width_px, height=15, color=(255,255,255), thickness=1)
#         rr, cc = circle_perimeter(self.best_y, self.best_x, 5)
#         qual_img[rr, cc] = 255
#         return pos_img, ang_img, width_img, qual_img

#     def depth_process_ggcnn(self):
#         depth_message = self.latest_depth_message
#         if depth_message is None or self.color_img is None:
#             return

#         depth = self.bridge.imgmsg_to_cv2(depth_message, "16UC1")
#         depth = depth.astype(np.float32)  
#         depth_copy_for_point_depth = depth.copy()
        
#         height_res, width_res = depth.shape
        
#         # MANTÉM O CROP ORIGINAL ESTÁVEL!
#         offset_x = (width_res - self.crop_size)//2
#         offset_y = 0
#         depth_crop = depth[offset_y : offset_y + self.crop_size, offset_x : offset_x + self.crop_size]
#         depth_crop = depth_crop.copy()
        
#         depth_nan = np.isnan(depth_crop)
#         depth_crop[depth_nan] = 0

#         mask = (depth_crop == 0).astype(np.uint8)
#         depth_scale = np.abs(depth_crop).max()
#         depth_crop = depth_crop.astype(np.float32) / depth_scale 
#         depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS) 
#         depth_crop = depth_crop[1:-1, 1:-1]
#         depth_crop = depth_crop * depth_scale

#         depth_crop = depth_crop/1000.0
#         depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
#         depth_tensor = torch.from_numpy(depth_crop).unsqueeze(0).unsqueeze(0) 
        
#         self.model.eval() 
#         with torch.no_grad(): 
#             pred_out = self.model(depth_tensor)  
        
#         points_out = pred_out[0].squeeze().cpu().numpy()
#         cos_out = pred_out[1].squeeze().cpu().numpy()
#         sin_out = pred_out[2].squeeze().cpu().numpy()
#         ang_out = np.arctan2(sin_out, cos_out) / 2.0  
#         width_out = pred_out[3].squeeze().cpu().numpy() * 150 
        
#         pos_out_filtered = cv2.GaussianBlur(points_out, (5, 5), 0) 
#         pos_out_filtered = np.clip(pos_out_filtered, 0.0, 1.0 - 1e-3)
#         ang_out_filtered = cv2.GaussianBlur(ang_out, (5, 5), 0)
#         width_out_filtered = cv2.GaussianBlur(width_out, (5, 5), 0)
            
#         # ==========================================================
#         # MÁSCARA COM PROJEÇÃO DE CAMPO TOTAL (DEBUG)
#         # ==========================================================
#         mask_2d = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        
#         # Criamos uma imagem de diagnóstico do tamanho da imagem original da câmera
#         debug_view = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

#         if hasattr(self, 'unity_target_base_link') and self.unity_target_base_link is not None:
#             try:
#                 pt_stamped = tf2_geometry_msgs.PointStamped()
#                 pt_stamped.header.frame_id = "base_link"
#                 pt_stamped.point.x = self.unity_target_base_link[0]
#                 pt_stamped.point.y = self.unity_target_base_link[1]
#                 pt_stamped.point.z = self.unity_target_base_link[2]
                
#                 cam_frame = depth_message.header.frame_id
#                 pt_cam = self.tf_buffer.transform(pt_stamped, cam_frame, rospy.Duration(0.2))
                
#                 if pt_cam.point.z > 0:
#                     u = int((self.fx * pt_cam.point.x) / pt_cam.point.z + self.cx)
#                     v = int((self.fy * pt_cam.point.y) / pt_cam.point.z + self.cy)
                    
#                     # Desenha um círculo na imagem COMPLETA para sabermos onde o ponto caiu
#                     cv2.circle(debug_view, (u, v), 10, (0, 255, 0), -1) # Verde: Ponto projetado
                    
#                     u_crop = u - offset_x
#                     v_crop = v - offset_y
                    
#                     if 0 <= u_crop < self.crop_size and 0 <= v_crop < self.crop_size:
#                         cv2.circle(mask_2d, (u_crop, v_crop), 45, 1.0, -1)
#                         pos_out_filtered = pos_out_filtered * mask_2d
#                         rospy.loginfo_throttle(1.0, "[GGCNN] DENTRO DO CROP: Mascara aplicada!")
#                     else:
#                         # Se estiver fora do crop, desenha um X vermelho na visão de debug
#                         cv2.line(debug_view, (u-20, v-20), (u+20, v+20), (0,0,255), 3)
#                         cv2.line(debug_view, (u+20, v-20), (u-20, v+20), (0,0,255), 3)
#                         rospy.logwarn_throttle(2.0, f"[GGCNN] FORA DO CROP: U={u_crop}, V={v_crop}")
                        
#             except Exception as e:
#                 rospy.logwarn_throttle(2.0, f"[TF ERROR] {e}")

#         # Mostra a visão de diagnóstico (Imagem da câmera + ponto projetado)
#         cv2.imshow("Debug: Projecao de Intencao", debug_view)
#         cv2.imshow("Debug: Mascara 300x300", mask_2d)
#         cv2.waitKey(1)
#         # ==========================================================

#         try:
#             transform_stamped = self.tf_buffer.lookup_transform("base_link", "robotiq_arg2f_base_link", rospy.Time(0), rospy.Duration(1.0))
#             ROBOT_Z = transform_stamped.transform.translation.z
#         except:
#             ROBOT_Z = 0.0
        
#         max_pixel = np.array(np.unravel_index(np.argmax(pos_out_filtered), pos_out_filtered.shape)) 
#         grasp_quality = pos_out_filtered[max_pixel[0], max_pixel[1]]   
        
#         # Se a máscara zerou tudo (clique ruim), aborta para a garra não voar para a origem
#         if grasp_quality < 0.001:
#             return

#         self.best_y, self.best_x = max_pixel.astype(int)
#         ang = ang_out_filtered[max_pixel[0], max_pixel[1]]   
#         width_px = abs(width_out_filtered[max_pixel[0], max_pixel[1]])  
        
#         reescaled_height = int(max_pixel[0]) 
#         reescaled_width = int(offset_x + max_pixel[1])
#         max_pixel_reescaled = [reescaled_height, reescaled_width]
#         point_depth = depth_copy_for_point_depth[max_pixel_reescaled[0], max_pixel_reescaled[1]] 

#         g_width = 2.0 * (ROBOT_Z + 0.24) * np.tan(self.FOV / height_res * width_px / 2.0 / 180.0 * np.pi) 
#         crop_size_width = float(self.crop_size)
#         width_m = width_out_filtered / crop_size_width * 2.0 * point_depth * np.tan(self.FOV * crop_size_width / height_res / 2.0 / 180.0 * np.pi) / 1000 
#         width_m = abs(width_m[max_pixel[0], max_pixel[1]])

#         if not np.isnan(point_depth):
#             x = (max_pixel_reescaled[1] - self.cx)/(self.fx) * point_depth 
#             y = (max_pixel_reescaled[0] - self.cy)/(self.fy) * point_depth
#             grasping_point = [x, y, point_depth] 

#             self.ang_out = ang_out
#             self.width_out = width_out
#             self.points_out = points_out
#             self.depth_message_ggcnn = depth_message
#             self.depth_crop = depth_crop
#             self.ang = ang
#             self.width_px = width_px
#             self.max_pixel = max_pixel
#             self.max_pixel_reescaled = max_pixel_reescaled
#             self.g_width = g_width
#             self.width_m = width_m
#             self.point_depth = point_depth
#             self.grasping_point = grasping_point
#             self.qual_out = grasp_quality   
#             self.pos_out_filtered = pos_out_filtered  

#     def publish_images(self):
#         if not hasattr(self, 'points_out') or self.points_out is None:
#             return
        
#         pos_img, ang_img, width_img, qual_img = self._generate_visual_maps(
#             self.points_out, self.ang_out, self.width_out, self.pos_out_filtered
#         )
#         pos_msg = self.bridge.cv2_to_imgmsg(pos_img, 'bgr8')
#         pos_msg.header = self.depth_message_ggcnn.header
#         self.grasp_pub.publish(pos_msg)

#         ang_msg = self.bridge.cv2_to_imgmsg(ang_img, 'bgr8')
#         ang_msg.header = self.depth_message_ggcnn.header
#         self.ang_pub.publish(ang_msg)

#         width_msg = self.bridge.cv2_to_imgmsg(width_img, 'bgr8')
#         width_msg.header = self.depth_message_ggcnn.header
#         self.width_pub.publish(width_msg)
        
#         qual_msg = self.bridge.cv2_to_imgmsg(qual_img, 'bgr8')
#         qual_msg.header = self.depth_message_ggcnn.header
#         self.depth_pub.publish(qual_msg)

#     def publish_data_to_robot(self):
#         if not hasattr(self, 'grasping_point') or not self.grasping_point:
#             return

#         cmd_msg = Float32MultiArray()
#         cmd_msg.data = [self.grasping_point[0]/1000.0, self.grasping_point[1]/1000.0, self.grasping_point[2]/1000.0, -1*self.ang, self.width_m, self.g_width]
#         self.cmd_pub.publish(cmd_msg)
        
#         grasp_transform = TransformStamped()
#         grasp_transform.header.stamp = rospy.Time.now()
#         grasp_transform.header.frame_id = "camera_depth_optical_frame"
#         grasp_transform.child_frame_id = "object_detected"
#         grasp_transform.transform.translation.x = cmd_msg.data[0]
#         grasp_transform.transform.translation.y = cmd_msg.data[1]
#         grasp_transform.transform.translation.z = cmd_msg.data[2]
#         q = quaternion_from_euler(3.14, 0, -1*cmd_msg.data[3])
#         grasp_transform.transform.rotation.x = q[0]
#         grasp_transform.transform.rotation.y = q[1]
#         grasp_transform.transform.rotation.z = q[2]
#         grasp_transform.transform.rotation.w = q[3]

#         self.tf_broadcaster.sendTransform(grasp_transform)

#     def get_transform_between_frames(self, target_frame, source_frame):
#         try:
#             transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
            
#             x = transform.transform.translation.x
#             y = transform.transform.translation.y
#             z = transform.transform.translation.z   
            
#             roll = 3.140
#             pitch = 0.0
#             yaw = -1 * self.ang  
#             quat = quaternion_from_euler(roll, pitch, yaw)
            
#             unity_pose = PoseStamped()
#             unity_pose.header.stamp = rospy.Time.now()
#             unity_pose.header.frame_id = target_frame 
#             unity_pose.pose.position.x = x
#             unity_pose.pose.position.y = y
#             unity_pose.pose.position.z = z
#             unity_pose.pose.orientation.x = quat[0]
#             unity_pose.pose.orientation.y = quat[1]
#             unity_pose.pose.orientation.z = quat[2]
#             unity_pose.pose.orientation.w = quat[3]

#             self.unity_pose_pub.publish(unity_pose)
#             self.unity_width_pub.publish(self.width_m)

#             cmd_msg_grasp = Float32MultiArray()
#             cmd_msg_grasp.data = [x, y, z, self.ang, self.width_m, self.g_width]
#             self.cmd_pub_grasp.publish(cmd_msg_grasp)

#             self.publish_static_transform(x, y, z, roll, pitch, yaw, 'base_link', 'object_grasp')
#             return transform
#         except:
#             return None

#     def publish_static_transform(self, x, y, z, roll, pitch, yaw, parent_frame, child_frame):
#         tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
#         static_transform_stamped = TransformStamped()
#         static_transform_stamped.header.stamp = rospy.Time.now()
#         static_transform_stamped.header.frame_id = parent_frame
#         static_transform_stamped.child_frame_id = child_frame
#         static_transform_stamped.transform.translation.x = x
#         static_transform_stamped.transform.translation.y = y
#         static_transform_stamped.transform.translation.z = z
#         quat = quaternion_from_euler(roll, pitch, yaw)
#         static_transform_stamped.transform.rotation.x = quat[0]
#         static_transform_stamped.transform.rotation.y = quat[1]
#         static_transform_stamped.transform.rotation.z = quat[2]
#         static_transform_stamped.transform.rotation.w = quat[3]
#         tf_broadcaster.sendTransform(static_transform_stamped)

# def parse_args():
#     parser = argparse.ArgumentParser(description='GGCNN grasping')
#     parser.add_argument('--real', action='store_true')
#     parser.add_argument('--plot', action='store_true')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     grasp_detection = ggcnn_grasping(args)
#     rospy.sleep(1.0)
#     print("Iniciando processo GGCNN...")
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         grasp_detection.depth_process_ggcnn()
#         grasp_detection.publish_images()
#         grasp_detection.publish_data_to_robot()
#         grasp_detection.get_transform_between_frames("base_link", "object_detected")
#         rate.sleep()

# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass