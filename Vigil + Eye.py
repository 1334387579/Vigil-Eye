import cv2
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import time
from PIL import Image, ImageDraw, ImageFont
import platform
import math

def add_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    """在图像上添加中文文本"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    try:
        fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    except:
        fontStyle = ImageFont.load_default()
    draw.text(position, text, textColor, font=fontStyle)
    
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 检测GPU可用性并设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device.upper()}")

# 初始化MediaPipe面部网格
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 加载YOLOv10模型并自动选择设备
model = YOLO('yolov10n.pt').to(device)

# 定义颜色常量
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 定义疲劳参数
EYE_AR_THRESH = 0.21  # 眼睛纵横比阈值
EYE_AR_CONSEC_FRAMES = 3  # 连续帧数
YAWN_THRESH = 0.55  # 打哈欠阈值
HEAD_TILT_THRESH = 25  # 头部倾斜阈值(度)
HEAD_PITCH_THRESH = 20  # 点头阈值(度)
HEAD_YAW_THRESH = 30  # 摇头阈值(度)
BLINK_RATE_THRESH = 15  # 眨眼频率阈值(次/分钟)
YAWN_RATE_THRESH = 3  # 打哈欠频率阈值(次/分钟)
DROWSY_TIME_THRESH = 2  # 疲劳状态持续时间阈值(秒)
MAR_SMOOTH_WINDOW = 5  # MAR平滑窗口大小
HEAD_POSE_SMOOTH_WINDOW = 5  # 头部角度平滑窗口
CONFIDENCE_THRESH = 0.5  # 降低关键点置信度阈值

# 初始化队列和计数器
eye_ar_history = deque(maxlen=EYE_AR_CONSEC_FRAMES)
blink_counter = 0
total_blinks = 0
yawn_counter = 0
drowsy_start_time = None
start_time = time.time()
last_yawn_time = 0
last_blink_time = 0
mar_history = deque(maxlen=MAR_SMOOTH_WINDOW)
head_pose_history = deque(maxlen=HEAD_POSE_SMOOTH_WINDOW)

# 定义面部关键点索引
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # 左眼轮廓关键点
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # 右眼轮廓关键点

# 嘴部关键点
MOUTH_OUTER_INDICES = [
    61,   # 左嘴角
    291,  # 右嘴角
    0,    # 上唇中点
    17,   # 下唇中点
    37,   # 上唇右点
    84,   # 下唇右点
    267,  # 上唇左点
    314   # 下唇左点
]

# 头部姿态估计关键点(使用更稳定的关键点组合)
POSE_INDICES = [
    1,     # 鼻根
    152,   # 下巴
    33,    # 左眼角
    263,   # 右眼角
    61,    # 左嘴角
    291,   # 右嘴角
    168,   # 左眉中(更稳定)
    397    # 右眉中(更稳定)
]

def eye_aspect_ratio(eye_points):
    """改进的眼睛纵横比(EAR)计算"""
    # 计算垂直距离
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # 计算水平距离
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # 计算EAR
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

def mouth_aspect_ratio(mouth_points):
    """改进的嘴部纵横比计算"""
    if len(mouth_points) < 8:
        return 0.0
    
    # 计算嘴部高度
    height1 = np.linalg.norm(mouth_points[2] - mouth_points[3])
    height2 = np.linalg.norm(mouth_points[4] - mouth_points[5])
    height3 = np.linalg.norm(mouth_points[6] - mouth_points[7])
    
    # 计算嘴部宽度
    width = np.linalg.norm(mouth_points[0] - mouth_points[1])
    
    avg_height = (height1 + height2 + height3) / 3.0
    mar = avg_height / (width + 1e-6)
    return mar

# def get_head_pose(face_landmarks, frame_size):
#     """改进的头部姿态估计"""
#     if len(face_landmarks) < 468:
#         return [0, 0, 0], False
    
#     # 3D模型点(单位:毫米)
#     model_points = np.array([
#         (0.0, 0.0, 0.0),         # 鼻根
#         (0.0, -330.0, -65.0),     # 下巴
#         (-165.0, 170.0, -135.0),  # 左眼角
#         (165.0, 170.0, -135.0),   # 右眼角
#         (-150.0, -150.0, -125.0), # 左嘴角
#         (150.0, -150.0, -125.0),  # 右嘴角
#         (-130.0, 70.0, -135.0),   # 左眉中(更稳定)
#         (130.0, 70.0, -135.0)     # 右眉中(更稳定)
#     ], dtype=np.float64)
    
#     # 2D图像点
#     image_points = []
#     for idx in POSE_INDICES:
#         image_points.append([face_landmarks[idx].x * frame_size[1], 
#                            face_landmarks[idx].y * frame_size[0]])
    
#     image_points = np.array(image_points, dtype=np.float64)
    
#     # 相机内参
#     focal_length = frame_size[1]
#     center = (frame_size[1]/2, frame_size[0]/2)
#     camera_matrix = np.array([
#         [focal_length, 0, center[0]],
#         [0, focal_length, center[1]],
#         [0, 0, 1]
#     ], dtype=np.float64)
    
#     dist_coeffs = np.zeros((4,1))
    
#     # 求解PnP问题
#     success, rotation_vec, translation_vec = cv2.solvePnP(
#         model_points, image_points, camera_matrix, dist_coeffs, 
#         flags=cv2.SOLVEPNP_EPNP)  # 使用EPNP算法，更稳定
    
#     if not success:
#         return [0, 0, 0], False
    
#     # 计算旋转矩阵
#     rmat, _ = cv2.Rodrigues(rotation_vec)
    
#     # 计算欧拉角
#     angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
#     # 转换为角度并调整范围
#     pitch = np.clip(angles[0] * 180 / math.pi, -90, 90)
#     yaw = np.clip(-angles[1] * 180 / math.pi, -90, 90)  # 取反以符合直观
#     roll = np.clip(angles[2] * 180 / math.pi, -90, 90)
    
#     return [pitch, yaw, roll], True

def get_head_pose(face_landmarks, frame_size):
    """改进版头部姿态估计"""
    if len(face_landmarks) < 468:
        return [0, 0, 0], False
    
    # 合理缩放的3D模型点 (单位：适当比例)
    model_points = np.array([
        (0.0, 0.0, 0.0),        # 鼻根
        (0.0, -33.0, -6.5),     # 下巴
        (-16.5, 17.0, -13.5),   # 左眼角
        (16.5, 17.0, -13.5),    # 右眼角
        (-15.0, -15.0, -12.5),  # 左嘴角
        (15.0, -15.0, -12.5),   # 右嘴角
        (-13.0, 7.0, -13.5),    # 左眉中
        (13.0, 7.0, -13.5)      # 右眉中
    ], dtype=np.float64) / 10.0
    
    # 2D图像点
    image_points = []
    for idx in POSE_INDICES:
        image_points.append([
            face_landmarks[idx].x * frame_size[1], 
            face_landmarks[idx].y * frame_size[0]
        ])
    
    image_points = np.array(image_points, dtype=np.float64)
    
    # 相机内参
    focal_length = frame_size[1]
    center = (frame_size[1]/2, frame_size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4,1))
    
    # 求解PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)  # 使用迭代法更稳定
    
    if not success:
        return [0, 0, 0], False
    
    # 计算旋转矩阵
    rmat, _ = cv2.Rodrigues(rotation_vec)
    
    # 直接计算欧拉角
    pitch = np.arctan2(-rmat[2,0], np.sqrt(rmat[2,1]**2 + rmat[2,2]**2)) * 180/math.pi
    yaw = np.arctan2(rmat[1,0], rmat[0,0]) * 180/math.pi
    roll = np.arctan2(rmat[2,1], rmat[2,2]) * 180/math.pi
    
    # 角度范围限制
    pitch = np.clip(pitch, -90, 90)
    yaw = np.clip(yaw, -90, 90)
    roll = np.clip(roll, -90, 90)
    
    return [pitch, yaw, roll], True


def smooth_angles(angle_history):
    """简化的角度平滑"""
    if len(angle_history) == 0:
        return [0, 0, 0]
    
    # 使用简单移动平均
    return np.mean(angle_history, axis=0)

def draw_landmarks(frame, landmarks, color=GREEN, radius=1):
    """在帧上绘制面部关键点"""
    for point in landmarks:
        cv2.circle(frame, tuple(point.astype(int)), radius, color, -1)
    return frame

def calculate_blink_rate(blink_count, elapsed_time):
    """计算眨眼频率(次/分钟)"""
    return blink_count / (elapsed_time / 60) if elapsed_time > 0 else 0

def calculate_yawn_rate(yawn_count, elapsed_time):
    """计算打哈欠频率(次/分钟)"""
    return yawn_count / (elapsed_time / 60) if elapsed_time > 0 else 0


def process_frame(frame):
    global blink_counter, total_blinks, yawn_counter, drowsy_start_time, last_yawn_time, last_blink_time, mar_history, head_pose_history
    
    # 转换为RGB格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # 初始化检测结果
    fatigue_warning = ""
    drowsy_warning = ""
    distraction_warning = ""
    violation_warning = ""
    head_pose_status = ""
    eye_status = "眼睛: 正常"
    mouth_status = "嘴巴: 闭合"
    current_time = time.time()
    # 面部特征检测
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        landmarks = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks])
        
        # 眼睛检测
        left_eye = landmarks[LEFT_EYE_INDICES]
        right_eye = landmarks[RIGHT_EYE_INDICES]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        eye_ar_history.append(avg_ear)
        
        # 眨眼检测
        if len(eye_ar_history) >= EYE_AR_CONSEC_FRAMES and all(ear < EYE_AR_THRESH for ear in eye_ar_history):
            if current_time - last_blink_time > 0.3:
                blink_counter += 1
                total_blinks += 1
                last_blink_time = current_time
                eye_ar_history.clear()
        
        # 眼睛状态
        if avg_ear < EYE_AR_THRESH:
            eye_status = "眼睛: 闭合"
        else:
            eye_status = "眼睛: 睁开"
        
        # 嘴巴/哈欠检测
        mouth_outer = landmarks[MOUTH_OUTER_INDICES]
        mar = mouth_aspect_ratio(mouth_outer)
        mar_history.append(mar)
        smoothed_mar = np.median(list(mar_history)) if len(mar_history) >= MAR_SMOOTH_WINDOW else mar
        
        if smoothed_mar > YAWN_THRESH:
            if current_time - last_yawn_time > 2.0:
                yawn_counter += 1
                last_yawn_time = current_time
                mouth_status = "嘴巴: 打哈欠"
        elif smoothed_mar > YAWN_THRESH * 0.8:
            mouth_status = "嘴巴: 张开"
        else:
            mouth_status = "嘴巴: 闭合"
        
        # 头部姿态检测
        raw_angles, is_valid = get_head_pose(face_landmarks, frame.shape)
        
        if is_valid:
            head_pose_history.append(raw_angles)
            
            # 计算平滑后的角度
            if len(head_pose_history) > 0:
                smoothed_angles = smooth_angles(list(head_pose_history))
                head_pitch = smoothed_angles[0]
                head_yaw = smoothed_angles[1]
                head_roll = smoothed_angles[2]
                
                # 当倾斜超过阈值时添加警告信息
                if abs(head_roll) > HEAD_TILT_THRESH:
                    if head_roll > 0:
                        head_pose_status = " (头部左倾警告!)"
                    else:
                        head_pose_status = " (头部右倾警告!)"
                
                # 添加点头警告
                if head_pitch > HEAD_PITCH_THRESH:
                    head_pose_status += " (低头警告!)"
                    drowsy_warning = "打瞌睡警告!"
                
                # 添加摇头警告
                if abs(head_yaw) > HEAD_YAW_THRESH:
                    if head_yaw > 0:
                        head_pose_status += " (左偏警告!)"
                        distraction_warning = "注意力分散警告(左看)!"
                    else:
                        head_pose_status += " (右偏警告!)"
                        distraction_warning = "注意力分散警告(右看)!"
                
                # 绘制头部姿态轴线
                nose_tip = landmarks[1].astype(int)
                chin = landmarks[152].astype(int)
                cv2.line(frame, tuple(nose_tip), tuple(chin), GREEN, 2)
            
            # 疲劳状态判断
            elapsed_time = current_time - start_time
            blink_rate = calculate_blink_rate(total_blinks, elapsed_time)
            yawn_rate = calculate_yawn_rate(yawn_counter, elapsed_time)
            
            # 疲劳条件（新增pitch和yaw判断）
            fatigue_conditions = [
                blink_rate > BLINK_RATE_THRESH,
                yawn_rate > YAWN_RATE_THRESH,
                abs(head_roll) > HEAD_TILT_THRESH,
                head_pitch > HEAD_PITCH_THRESH,
                abs(head_yaw) > HEAD_YAW_THRESH,
                avg_ear < EYE_AR_THRESH * 1.2
            ]
            
            if any(fatigue_conditions):
                if drowsy_start_time is None:
                    drowsy_start_time = current_time
                elif current_time - drowsy_start_time > DROWSY_TIME_THRESH:
                    fatigue_warning = "疲劳警告!"
            else:
                drowsy_start_time = None
        
        # 绘制面部特征
        frame = draw_landmarks(frame, left_eye, BLUE)
        frame = draw_landmarks(frame, right_eye, BLUE)
        frame = draw_landmarks(frame, mouth_outer, YELLOW)
        
        # 显示EAR和MAR值
        frame = add_chinese_text(frame, f"EAR: {avg_ear:.2f}", (10, 30), GREEN, 20)
        frame = add_chinese_text(frame, f"MAR: {smoothed_mar:.2f}", (10, 60), YELLOW, 20)
        frame = add_chinese_text(frame, eye_status, (10, 90), RED, 15)
        frame = add_chinese_text(frame, mouth_status, (10, 120), RED, 15)
    
    # YOLO违规行为检测
    yolo_results = model(frame, verbose=False)
    for result in yolo_results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            label = model.names[class_id]
            
            if label in ['cell phone', 'cup', 'bottle', 'sandwich', 'person'] and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
                frame = add_chinese_text(frame, f"{label}: {confidence:.2f}", (x1, y1-30), RED, 15)
                
                if label == 'cell phone':
                    violation_warning = "使用手机警告!"
                elif label in ['cup', 'bottle']:
                    violation_warning = "饮食警告!"
                elif label == 'sandwich':
                    violation_warning = "进食警告!"
    
    # 显示警告信息
    warning_y_offset = 30
    if fatigue_warning:
        frame = add_chinese_text(frame, fatigue_warning, (frame.shape[1]//2-100, warning_y_offset), RED, 25)
        warning_y_offset += 30
    if drowsy_warning:
        frame = add_chinese_text(frame, drowsy_warning, (frame.shape[1]//2-100, warning_y_offset), RED, 25)
        warning_y_offset += 30
    if distraction_warning:
        frame = add_chinese_text(frame, distraction_warning, (frame.shape[1]//2-100, warning_y_offset), RED, 25)
        warning_y_offset += 30
    if violation_warning:
        frame = add_chinese_text(frame, violation_warning, (frame.shape[1]//2-100, warning_y_offset), RED, 25)
    
    # 显示统计信息和头部姿态信息（统一在左侧）
    elapsed_time = current_time - start_time
    blink_rate = calculate_blink_rate(total_blinks, elapsed_time)
    yawn_rate = calculate_yawn_rate(yawn_counter, elapsed_time)
    
    # 合并统计信息和头部姿态信息
    stat_texts = [
        f"眨眼次数: {total_blinks}",
        f"哈欠次数: {yawn_counter}",
        f"眨眼频率: {blink_rate:.1f}/min",
        f"哈欠频率: {yawn_rate:.1f}/min",
        f"头部俯仰: {head_pitch:>+6.1f}°" if 'head_pitch' in locals() else "头部俯仰: N/A",
        f"头部偏航: {head_yaw:>+6.1f}°" if 'head_yaw' in locals() else "头部偏航: N/A",
        f"头部旋转: {head_roll:>+6.1f}°" if 'head_roll' in locals() else "头部旋转: N/A"
    ]
    
    # 显示所有信息
    for i, stat in enumerate(stat_texts):
        frame = add_chinese_text(frame, stat, (10, 150 + i*30), RED, 15)
    
    # 显示头部姿态警告信息
    if head_pose_status:
        frame = add_chinese_text(frame, head_pose_status, (10, 150 + len(stat_texts)*30), RED, 15)
    
    return frame



def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 创建可调整大小的窗口
    window_name = 'Driver Fatigue Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL 允许调整窗口大小
    cv2.resizeWindow(window_name, 1200, 600)  # 设置初始窗口大小
    
    print("按ESC键退出程序...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break
            
            # 处理帧
            processed_frame = process_frame(frame)
            
            # 显示结果
            cv2.imshow(window_name, processed_frame)
            
            # 检查ESC键
            if cv2.waitKey(1) == 27:
                break
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序结束")


if __name__ == "__main__":
    main()
