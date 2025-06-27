#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于传统计算机视觉的Box积木识别系统

功能概述:
- 使用颜色分割和轮廓检测识别box1_model和box2_model
- 基于积木的几何特征进行分类
- 输出积木的3D位置和类型信息

技术架构:
- ROS: 机器人操作系统通信
- OpenCV: 计算机视觉处理
- RGB-D: 彩色和深度图像融合
- 几何分析: 基于尺寸比例的积木分类
"""

import rospy
import cv2 as cv
import numpy as np
import time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Pose, Quaternion
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge
import message_filters
from pyquaternion import Quaternion as PyQuaternion

# ========== 全局变量定义 ==========

# 调试显示开关
a_show = True

# 图像处理相关
img_draw = None
bridge = None
pub = None

# 工作台参数 (参考lego-vision.py的实际配置)
height_tavolo = 0.74  # 工作台高度
dist_tavolo = None    # 相机到工作台距离 (动态计算)
cam_point = (-0.44, -0.5, 1.58)  # 相机位置
origin = None  # 图像中心点 (动态计算)

# ========== 颜色定义和映射 ==========

# Gazebo材质颜色到HSV范围的映射
COLOR_RANGES = {
    'Gazebo/Red': {
        'hsv_lower': np.array([0, 100, 100]),
        'hsv_upper': np.array([10, 255, 255]),
        'name': 'red'
    },
    'Gazebo/Orange': {
        'hsv_lower': np.array([11, 100, 100]),
        'hsv_upper': np.array([25, 255, 255]),
        'name': 'orange'
    },
    'Gazebo/DarkYellow': {
        'hsv_lower': np.array([26, 100, 100]),
        'hsv_upper': np.array([35, 255, 255]),
        'name': 'yellow'
    },
    'Gazebo/Green': {
        'hsv_lower': np.array([36, 100, 100]),
        'hsv_upper': np.array([85, 255, 255]),
        'name': 'green'
    },
    'Gazebo/SkyBlue': {
        'hsv_lower': np.array([86, 100, 100]),
        'hsv_upper': np.array([125, 255, 255]),
        'name': 'blue'
    },
    'Gazebo/Indigo': {
        'hsv_lower': np.array([126, 100, 100]),
        'hsv_upper': np.array([145, 255, 255]),
        'name': 'indigo'
    },
    'Gazebo/Purple': {
        'hsv_lower': np.array([146, 100, 100]),
        'hsv_upper': np.array([170, 255, 255]),
        'name': 'purple'
    },
    'Gazebo/White': {
        'hsv_lower': np.array([0, 0, 200]),
        'hsv_upper': np.array([180, 30, 255]),
        'name': 'white'
    },
    'Gazebo/Gray': {
        'hsv_lower': np.array([0, 0, 50]),
        'hsv_upper': np.array([180, 30, 200]),
        'name': 'gray'
    }
}

# 积木类型定义
BOX_TYPES = {
    'box1_model': {
        'size': [0.025, 0.025, 0.02],  # 正方形积木
        'aspect_ratio_range': [0.7, 1.4],  # 放宽长宽比范围
        'min_area': 80,   # 降低最小面积阈值
        'max_area': 6000  # 提高最大面积阈值
    },
    'box2_model': {
        'size': [0.025, 0.01, 0.02],   # 长方形积木
        'aspect_ratio_range': [1.5, 4.0],  # 放宽长宽比范围
        'min_area': 40,   # 降低最小面积阈值
        'max_area': 4000  # 提高最大面积阈值
    }
}

# ========== 工具函数 ==========

def get_dist_tavolo(depth, hsv, img_draw):
    """
    计算相机到工作台的距离
    
    参数:
        depth: 深度图像数组
        hsv: HSV颜色空间图像
        img_draw: 用于绘制的图像
    
    功能:
        通过分析深度图像确定工作台的距离，这是后续3D定位的基础
    """
    global dist_tavolo
    dist_tavolo = np.nanmax(depth)

def get_origin(img):
    """
    获取图像的中心点坐标
    
    参数:
        img: 输入图像
    
    功能:
        计算图像的几何中心，用于后续的坐标变换
    """
    global origin
    origin = np.array(img.shape[1::-1]) // 2  # [width/2, height/2]

def get_depth_at_point(depth_image, x, y, radius=5):
    """
    获取指定点的深度值
    
    参数:
        depth_image: 深度图像
        x, y: 像素坐标
        radius: 采样半径
    
    返回:
        depth: 深度值(米)
    """
    h, w = depth_image.shape
    x, y = int(x), int(y)
    
    # 边界检查
    x = max(radius, min(w-radius-1, x))
    y = max(radius, min(h-radius-1, y))
    
    # 在半径范围内采样深度值
    depth_patch = depth_image[y-radius:y+radius+1, x-radius:x+radius+1]
    
    # 过滤无效深度值
    valid_depths = depth_patch[depth_patch > 0]
    
    if len(valid_depths) > 0:
        return np.median(valid_depths)  # 使用中位数减少噪声
    else:
        return 0.0

def pixel_to_world(x, y, depth):
    """
    将像素坐标转换为世界坐标 (参考lego-vision.py的精确算法)
    
    参数:
        x, y: 像素坐标
        depth: 深度值(米)
    
    返回:
        world_point: 世界坐标[x, y, z]
    """
    if depth <= 0:
        return None
    
    # 计算积木高度
    box_height = dist_tavolo - depth
    
    # 使用lego-vision.py的坐标变换算法
    xyz = np.array([x, y, box_height / 2 + height_tavolo])
    
    # 获取图像尺寸
    img_height, img_width = 480, 640  # 假设标准相机分辨率
    
    # 归一化像素坐标
    xyz[:2] /= img_width, img_height
    xyz[:2] -= 0.5
    
    # 应用相机视场角变换
    xyz[:2] *= (-0.968, 0.691)
    
    # 距离缩放
    xyz[:2] *= dist_tavolo / 0.84
    
    # 添加相机偏移
    xyz[:2] += cam_point[:2]
    
    return xyz

def point_distorption(point, height, origin):
    """
    应用透视畸变校正 (参考lego-vision.py)
    
    参数:
        point: 图像坐标点
        height: 物体高度
        origin: 图像中心点
    
    返回:
        校正后的图像坐标
    """
    p = dist_tavolo / (dist_tavolo - height)
    point = point - origin
    return p * point + origin

def point_inverse_distortption(point, height):
    """
    应用逆透视畸变校正 (参考lego-vision.py)
    
    参数:
        point: 已畸变的图像坐标点
        height: 物体高度
    
    返回:
        校正前的原始图像坐标
    """
    p = dist_tavolo / (dist_tavolo - height)
    point = point - origin
    return point / p + origin

def detect_color(hsv_image, mask_region):
    """
    检测积木颜色
    
    参数:
        hsv_image: HSV颜色空间图像
        mask_region: 积木区域掩码
    
    返回:
        color_name: 检测到的颜色名称
    """
    # 在积木区域内采样颜色
    masked_hsv = cv.bitwise_and(hsv_image, hsv_image, mask=mask_region)
    
    best_match = None
    max_pixels = 0
    
    # 遍历所有颜色范围
    for gazebo_color, color_info in COLOR_RANGES.items():
        # 创建颜色掩码
        color_mask = cv.inRange(masked_hsv, color_info['hsv_lower'], color_info['hsv_upper'])
        
        # 计算匹配像素数量
        pixel_count = cv.countNonZero(color_mask)
        
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            best_match = color_info['name']
    
    return best_match if max_pixels > 10 else 'unknown'

def classify_box_type(contour, aspect_ratio):
    """
    根据轮廓特征分类积木类型
    
    参数:
        contour: 轮廓
        aspect_ratio: 长宽比
    
    返回:
        box_type: 积木类型 ('box1_model' 或 'box2_model')
    """
    area = cv.contourArea(contour)
    
    # 检查box1_model (正方形)
    box1_info = BOX_TYPES['box1_model']
    if (box1_info['aspect_ratio_range'][0] <= aspect_ratio <= box1_info['aspect_ratio_range'][1] and
        box1_info['min_area'] <= area <= box1_info['max_area']):
        return 'box1_model'
    
    # 检查box2_model (长方形)
    box2_info = BOX_TYPES['box2_model']
    if (box2_info['aspect_ratio_range'][0] <= aspect_ratio <= box2_info['aspect_ratio_range'][1] and
        box2_info['min_area'] <= area <= box2_info['max_area']):
        return 'box2_model'
    
    return 'unknown'

def process_contour(contour, hsv_image, depth_image):
    """
    处理单个轮廓，提取积木信息
    
    参数:
        contour: 轮廓
        hsv_image: HSV图像
        depth_image: 深度图像
    
    返回:
        box_info: 积木信息字典或None
    """
    # 计算轮廓的最小外接矩形
    rect = cv.minAreaRect(contour)
    center, (width, height), angle = rect
    
    # 过滤过小的轮廓
    if width < 5 or height < 5:
        return None
    
    # 计算长宽比
    aspect_ratio = max(width, height) / min(width, height)
    
    # 分类积木类型
    box_type = classify_box_type(contour, aspect_ratio)
    if box_type == 'unknown':
        return None
    
    # 创建轮廓掩码
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [contour], 255)
    
    # 检测颜色
    color = detect_color(hsv_image, mask)
    if color == 'unknown':
        return None
    
    # 获取深度信息
    depth = get_depth_at_point(depth_image, center[0], center[1])
    if depth <= 0:
        return None
    
    # 计算积木高度
    box_height = dist_tavolo - depth
    
    # 应用透视畸变校正 (参考lego-vision.py)
    corrected_center = point_inverse_distortption(np.array(center), box_height)
    
    # 转换为世界坐标
    world_pos = pixel_to_world(corrected_center[0], corrected_center[1], depth)
    if world_pos is None:
        return None
    
    # 计算姿态 (简化为只考虑Z轴旋转)
    quaternion = PyQuaternion(axis=[0, 0, 1], angle=np.radians(angle))
    
    return {
        'type': box_type,
        'color': color,
        'position': world_pos,
        'orientation': quaternion,
        'center_pixel': center,
        'size_pixel': (width, height),
        'angle': angle
    }

def process_image(rgb_image, depth_image):
    """
    处理RGB-D图像，检测所有积木 (参考lego-vision.py的处理流程)
    
    参数:
        rgb_image: RGB图像
        depth_image: 深度图像
    
    返回:
        detections: 检测结果列表
    """
    global img_draw
    
    # ========== 图像预处理 ==========
    img_draw = rgb_image.copy()
    hsv_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2HSV)
    
    # ========== 工作台参数计算 ==========
    get_dist_tavolo(depth_image, hsv_image, img_draw)
    get_origin(rgb_image)
    
    # 创建前景掩码 (排除黑色背景)
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 30])
    
    # 创建非黑色区域掩码
    foreground_mask = cv.bitwise_not(cv.inRange(hsv_image, black_lower, black_upper))
    
    # 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_OPEN, kernel)
    foreground_mask = cv.morphologyEx(foreground_mask, cv.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv.findContours(foreground_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    # 处理每个轮廓
    for contour in contours:
        # 过滤过小轮廓
        area = cv.contourArea(contour)
        if area < 10:
            continue
            
        box_info = process_contour(contour, hsv_image, depth_image)
        
        if box_info is not None:
            detections.append(box_info)
            
            # 绘制调试信息 (参考lego-vision.py的可视化风格)
            if a_show:
                # 绘制轮廓
                cv.drawContours(img_draw, [contour], -1, (0, 255, 0), 2)
                
                # 绘制中心点
                center = tuple(map(int, box_info['center_pixel']))
                cv.circle(img_draw, center, 5, (255, 0, 0), -1)
                
                # 绘制最小外接矩形
                rect = cv.minAreaRect(contour)
                box_points = cv.boxPoints(rect)
                box_points = np.int0(box_points)
                cv.drawContours(img_draw, [box_points], 0, (0, 0, 255), 2)
                
                # 绘制坐标轴 (简化版)
                axis_length = 30
                # X轴 (红色)
                end_x = (center[0] + axis_length, center[1])
                cv.line(img_draw, center, end_x, (0, 0, 255), 1)
                cv.putText(img_draw, 'X', end_x, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Y轴 (绿色)
                end_y = (center[0], center[1] + axis_length)
                cv.line(img_draw, center, end_y, (0, 255, 0), 1)
                cv.putText(img_draw, 'Y', end_y, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 绘制详细文本信息
                text = f"{box_info['type']} {box_info['color']}"
                pos_text = f"({box_info['position'][0]:.3f}, {box_info['position'][1]:.3f}, {box_info['position'][2]:.3f})"
                
                # 创建文本背景
                (text_width, text_height) = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, 0.4, 1)[0]
                text_offset_x = center[0] - text_width // 2
                text_offset_y = center[1] - 40
                
                # 绘制文本背景
                box_coords = ((text_offset_x - 2, text_offset_y + 2), 
                             (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
                cv.rectangle(img_draw, box_coords[0], box_coords[1], (210, 210, 10), cv.FILLED)
                
                # 绘制文本
                cv.putText(img_draw, text, (text_offset_x, text_offset_y), 
                          cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
                cv.putText(img_draw, pos_text, (text_offset_x, text_offset_y + 15), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return detections

def create_model_states_msg(detections):
    """
    创建ROS ModelStates消息
    
    参数:
        detections: 检测结果列表
    
    返回:
        msg: ModelStates消息
    """
    msg = ModelStates()
    
    for i, detection in enumerate(detections):
        # 生成唯一名称
        model_name = f"{detection['type']}"
        msg.name.append(model_name)
        
        # 创建位置和姿态
        position = Point(
            x=detection['position'][0],
            y=detection['position'][1], 
            z=detection['position'][2]
        )
        
        orientation = Quaternion(
            x=detection['orientation'].x, # 长方体识别位姿在rotx存在一定偏差，待优化
            y=detection['orientation'].y,
            z=-detection['orientation'].z, # 取反
            w=detection['orientation'].w
        )
        
        pose = Pose(position=position, orientation=orientation)
        msg.pose.append(pose)
    
    return msg

def image_callback(rgb_msg, depth_msg):
    """
    ROS图像回调函数 (参考lego-vision.py的处理流程)
    
    参数:
        rgb_msg: RGB图像消息
        depth_msg: 深度图像消息
    """
    global bridge, pub, img_draw
    
    t_start = time.time()
    
    try:
        # ========== ROS图像消息转换 ==========
        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        
        # ========== 调用图像处理函数 ==========
        detections = process_image(rgb_image, depth_image)
        
        # ========== 创建并发布ROS消息 ==========
        if detections:
            msg = create_model_states_msg(detections)
            pub.publish(msg)
            
            rospy.loginfo(f"检测到 {len(detections)} 个积木")
            for detection in detections:
                rospy.loginfo(f"  - {detection['type']} ({detection['color']}) at {detection['position']}")
        else:
            rospy.loginfo("未检测到积木")
        
        # ========== 性能统计 ==========
        processing_time = time.time() - t_start
        print(f"处理时间: {processing_time:.3f} 秒")
        
        # ========== 调试显示 ==========
        if a_show and img_draw is not None:
            # 在图像上显示处理时间和检测数量
            info_text = f"Detected num: {len(detections)}, processing time: {processing_time:.3f}s"
            cv.putText(img_draw, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv.imshow('Box Detection Results', img_draw)
            cv.waitKey(1)
            
    except Exception as e:
        rospy.logerr(f"图像处理错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    主函数：初始化ROS节点和视觉识别系统 (参考lego-vision.py的启动流程)
    """
    global bridge, pub
    
    print("启动Box视觉识别节点 Box-Vision 1.0")
    
    # ========== 初始化ROS节点 ==========
    rospy.init_node('box_vision', anonymous=True)
    rospy.loginfo("Box视觉识别节点已启动")
    
    # ========== 创建CvBridge ==========
    bridge = CvBridge()
    
    # ========== 创建结果发布器 ==========
    pub = rospy.Publisher('box_detections', ModelStates, queue_size=1)
    
    print("订阅相机图像话题")
    # ========== 创建图像订阅器 ==========
    rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
    
    print("Box检测系统正在启动...")
    print("(等待图像数据...)", end='\r'), print(end='\033[K')  # 清除行末内容
    
    # ========== 图像同步机制 ==========
    # 使用TimeSynchronizer确保RGB和深度图像严格同步
    syncro = message_filters.TimeSynchronizer([rgb_sub, depth_sub], 1, reset=True)
    syncro.registerCallback(image_callback)
    
    rospy.loginfo("等待图像数据...")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("节点被用户中断")
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断")
    finally:
        print("\n关闭所有OpenCV窗口")
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()