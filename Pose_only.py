import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
from filterpy.kalman import KalmanFilter
import pandas as pd


class PoseKalmanFilter:
    def __init__(self, dim_x=6, dim_z=6, dim_u=0):
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z, dim_u=dim_u)
        self.kf.F = np.eye(dim_x)
        self.kf.H = np.eye(dim_z, dim_x)
        self.kf.R = np.eye(dim_z) * 0.1
        self.kf.Q = np.eye(dim_x) * 0.01
        self.kf.P *= 1000.

    def update(self, z):
        self.kf.update(z)

    def predict(self):
        self.kf.predict()
        return self.kf.x


def euler_to_rotation_matrix(pitch, yaw, roll):
    pitch, yaw, roll = map(float, (pitch, yaw, roll))
    pitch, yaw, roll = map(np.radians, (pitch, yaw, roll))

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])

    R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]])

    R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 获取相机内参
profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
color_intrinsics = color_profile.get_intrinsics()

# 构建相机矩阵
fx, fy = color_intrinsics.fx, color_intrinsics.fy
cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


# 相机内置的畸变系数
dist_coeffs = np.array([-0.001, -0.001, -0.023, 0], dtype=np.float32)  # 添加第四个参数为0


# 加载YOLOv8模型
model_path = r"D:\desktop_backup\Hot swappable UAV & Station\BPD_code\best_pose_yolov8n.pt"  # 请替换为您的模型路径
model = YOLO(model_path)

# 定义3D关键点（单位：米）
object_points = np.array([
    [-0.145, 0.04, 0],  # 左上角 (UL)
    [-0.145, -0.04, 0],  # 右上角 (UR)
    [0.145, -0.04, 0],  # 左下角 (LL)
    [0.145, 0.04, 0],  # 右下角 (LR)
    [0, 0.08, 0]  # 突出部分中点 (H)
], dtype=np.float32)

# 初始化卡尔曼滤波器
pose_filter = PoseKalmanFilter(dim_x=6, dim_z=6)  # 6 for position and rotation

# 多帧融合参数
frame_buffer = []
buffer_size = 20


def detect_keypoints(image, conf_thres=0.8):
    results = model(image, task='pose', conf=conf_thres)
    return results


def estimate_pose(keypoints_2d, object_points, camera_matrix, dist_coeffs):
    if len(keypoints_2d) != len(object_points):
        print(f"警告：关键点数量 ({len(keypoints_2d)}) 与 3D 点数量 ({len(object_points)}) 不匹配")
        return None, None

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            keypoints_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            return rvec, tvec
        else:
            print("姿态估计失败")
            return None, None
    except cv2.error as e:
        print(f"OpenCV 错误：{e}")
        return None, None
    except Exception as e:
        print(f"未知错误：{e}")
        return None, None


def visualize_pose(image, rvec, tvec, camera_matrix, dist_coeffs):
    axis_length = 0.1  # 10cm
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    origin = tuple(map(int, imgpts[0].ravel()))
    image = cv2.line(image, origin, tuple(map(int, imgpts[1].ravel())), (0, 0, 255), 3)  # X轴 红色
    image = cv2.line(image, origin, tuple(map(int, imgpts[2].ravel())), (0, 255, 0), 3)  # Y轴 绿色
    image = cv2.line(image, origin, tuple(map(int, imgpts[3].ravel())), (255, 0, 0), 3)  # Z轴 蓝色

    return image


def visualize_keypoints(color_image, depth_image, results):
    global frame_buffer

    annotated_image = color_image.copy()
    h, w = annotated_image.shape[:2]

    if results is None or len(results) == 0:
        print("未检测到物体")
        return annotated_image

    result = results[0]  # 假设只检测一个物体
    boxes = result.boxes
    keypoints = result.keypoints

    if boxes is None or len(boxes) == 0:
        print("未检测到边界框")
        return annotated_image

    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        print("未检测到关键点")
        return annotated_image

    # 绘制边界框
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 处理关键点
    kpts = keypoints.xy[0]
    conf = keypoints.conf
    if conf is not None and conf.numel() > 0:
        mean_conf = conf.mean().item()
        print(f"平均置信度: {mean_conf:.2f}")
        if mean_conf > 0.75:  # 添加置信度阈值
            valid_kpts = []
            keypoint_labels = ['UL', 'LL', 'LR', 'UR', 'H']  # 关键点标签
            for i, (x, y) in enumerate(kpts):
                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    valid_kpts.append([x, y])
                    cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
                    depth = depth_image[y, x]

                    # 添加关键点标签
                    label = keypoint_labels[i] if i < len(keypoint_labels) else f'K{i}'
                    cv2.putText(annotated_image, label, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    cv2.putText(annotated_image, f"{depth}mm", (x + 10, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    print(f"警告：关键点 {i} ({x}, {y}) 超出图像范围")

            if len(valid_kpts) == 5:
                rvec, tvec = estimate_pose(np.array(valid_kpts, dtype=np.float32), object_points, camera_matrix,
                                           dist_coeffs)
                if rvec is not None and tvec is not None:
                    # 将旋转向量转换为欧拉角
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, np.zeros((3, 1)))))[6]

                    # 构建状态向量
                    state = np.concatenate([tvec.flatten(), euler_angles.flatten()])

                    # 更新卡尔曼滤波器
                    pose_filter.update(state)

                    # 预测下一个状态
                    predicted_state = pose_filter.predict()

                    # 将预测的状态转换回rvec和tvec
                    predicted_tvec = predicted_state[:3].reshape(3, 1)
                    predicted_euler_angles = predicted_state[3:]
                    predicted_rotation_matrix = euler_to_rotation_matrix(*predicted_euler_angles)
                    predicted_rvec, _ = cv2.Rodrigues(predicted_rotation_matrix)

                    # 添加到帧缓冲
                    frame_buffer.append((predicted_rvec, predicted_tvec))
                    if len(frame_buffer) > buffer_size:
                        frame_buffer.pop(0)

                    # 计算多帧平均
                    avg_rvec = np.mean([f[0] for f in frame_buffer], axis=0)
                    avg_tvec = np.mean([f[1] for f in frame_buffer], axis=0)

                    # 将平均旋转向量转换为欧拉角
                    avg_rotation_matrix, _ = cv2.Rodrigues(avg_rvec)
                    avg_euler_angles = \
                        cv2.decomposeProjectionMatrix(np.hstack((avg_rotation_matrix, np.zeros((3, 1)))))[6]

                    # 输出 XYZ position 和 pose angle
                    print(f"XYZ Position: {avg_tvec.flatten()}")
                    print(f"Pose Angles (degrees): {avg_euler_angles.flatten()}")

                    # 可视化平均姿态
                    annotated_image = visualize_pose(annotated_image, avg_rvec, avg_tvec, camera_matrix, dist_coeffs)

                    # 在图像上显示 XYZ 和角度信息
                    cv2.putText(annotated_image,
                                f"X: {avg_tvec[0][0]:.2f}, Y: {avg_tvec[1][0]:.2f}, Z: {avg_tvec[2][0]:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_image,
                                f"Roll: {avg_euler_angles[0][0]:.2f}, Pitch: {avg_euler_angles[1][0]:.2f}, Yaw: {avg_euler_angles[2][0]:.2f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                else:
                    print("姿态估计失败")
            else:
                print(f"警告：检测到的有效关键点数量不足 (检测到 {len(valid_kpts)}, 需要 5)")
        else:
            print(f"关键点置信度过低: {mean_conf:.2f}")
    else:
        print("未检测到关键点置信度")

    return annotated_image


def record_and_analyze_data(num_samples=20):
    data = []
    total_processing_time = 0
    start_time = time.time()

    print("Recording data... Press 'q' to stop early.")

    while len(data) < num_samples:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Measure total processing time
        process_start = time.time()

        # Add your preprocessing steps here if any

        results = detect_keypoints(color_image)
        annotated_image = visualize_keypoints(color_image, depth_image, results)

        process_end = time.time()
        total_processing_time += process_end - process_start

        cv2.imshow('Recording Data', annotated_image)

        if len(frame_buffer) > 0:
            avg_rvec, avg_tvec = frame_buffer[-1]
            avg_rotation_matrix, _ = cv2.Rodrigues(avg_rvec)
            avg_euler_angles = cv2.decomposeProjectionMatrix(np.hstack((avg_rotation_matrix, np.zeros((3, 1)))))[6]

            timestamp = time.time() - start_time
            xyz = avg_tvec.flatten()
            angles = avg_euler_angles.flatten()

            data.append({
                'timestamp': timestamp,
                'x': xyz[0], 'y': xyz[1], 'z': xyz[2],
                'roll': angles[0], 'pitch': angles[1], 'yaw': angles[2]
            })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow('Recording Data')

    if len(data) == 0:
        print("No data recorded.")
        return

    df = pd.DataFrame(data)

    # Define true values
    true_values_position = [-0.01, -0.074, 0.665]  # Assuming Z is 0.4 meters
    true_values_attitude = [-178, 38, 90]  # Assuming Roll is -180 degrees

    # Calculate errors
    errors_position = {
        'x': df['x'] - true_values_position[0],
        'y': df['y'] - true_values_position[1],
        'z': df['z'] - true_values_position[2]
    }
    errors_attitude = {
        'roll': df['roll'] - true_values_attitude[0],
        'pitch': df['pitch'] - true_values_attitude[1],
        'yaw': df['yaw'] - true_values_attitude[2]
    }

    # Function to calculate metrics
    def calculate_metrics(errors):
        mae = {k: np.mean(np.abs(v)) for k, v in errors.items()}
        rmse = {k: np.sqrt(np.mean(v ** 2)) for k, v in errors.items()}
        sd = {k: np.std(v) for k, v in errors.items()}
        return mae, rmse, sd

    # Calculate metrics for position and attitude
    mae_position, rmse_position, sd_position = calculate_metrics(errors_position)
    mae_attitude, rmse_attitude, sd_attitude = calculate_metrics(errors_attitude)

    # Calculate overall metrics
    overall_position = {
        'mae': np.mean(list(mae_position.values())),
        'rmse': np.sqrt(np.mean([v ** 2 for v in rmse_position.values()])),
        'sd': np.sqrt(np.mean([v ** 2 for v in sd_position.values()]))
    }
    overall_attitude = {
        'mae': np.mean(list(mae_attitude.values())),
        'rmse': np.sqrt(np.mean([v ** 2 for v in rmse_attitude.values()])),
        'sd': np.sqrt(np.mean([v ** 2 for v in sd_attitude.values()]))
    }

    # Calculate average processing time
    avg_processing_time = total_processing_time / num_samples

    # Prepare results
    results = [
        # Position metrics
        mae_position['x'], mae_position['y'], mae_position['z'], overall_position['mae'],
        rmse_position['x'], rmse_position['y'], rmse_position['z'], overall_position['rmse'],
        sd_position['x'], sd_position['y'], sd_position['z'], overall_position['sd'],
        # Attitude metrics
        mae_attitude['roll'], mae_attitude['pitch'], mae_attitude['yaw'], overall_attitude['mae'],
        rmse_attitude['roll'], rmse_attitude['pitch'], rmse_attitude['yaw'], overall_attitude['rmse'],
        sd_attitude['roll'], sd_attitude['pitch'], sd_attitude['yaw'], overall_attitude['sd'],
        # Processing time
        avg_processing_time
    ]

    # Convert results to a tab-separated string for easy copying
    results_string = "\t".join(f"{value:.6f}" for value in results)
    print("\nCopy the following line into your Excel sheet:")
    print(results_string)

    return results
try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 将图像转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 检测关键点
        results = detect_keypoints(color_image)

        # 可视化结果
        annotated_image = visualize_keypoints(color_image, depth_image, results)

        # 显示原始图像和标注后的图像
        cv2.imshow('RealSense Original', color_image)
        cv2.imshow('RealSense Pose Estimation', annotated_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            record_and_analyze_data()
            break

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()

finally:
    # 停止相机流
    pipeline.stop()
    cv2.destroyAllWindows()