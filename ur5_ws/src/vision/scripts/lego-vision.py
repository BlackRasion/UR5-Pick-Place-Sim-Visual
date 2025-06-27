#! /usr/bin/env python3

"""
乐高积木视觉识别模块

功能概述:
该模块是UR5机械臂乐高积木抓取系统的核心视觉组件，主要负责:
1. 从RGB-D相机获取彩色图像和深度图像
2. 使用YOLO模型进行积木检测和分类
3. 通过深度信息计算积木的3D位置
4. 确定积木的方向和姿态
5. 发布积木的位置和姿态信息供运动规划模块使用

技术架构:
- 基于ROS (Robot Operating System) 框架
- 使用YOLOv5进行目标检测
- 结合RGB和深度信息进行3D定位
- 采用计算机视觉技术进行姿态估计
"""

import cv2 as cv
import numpy as np
import torch
import message_filters
import rospy
import sys
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rospkg import RosPack # ROS包路径获取工具
from os import path # 文件路径操作
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import *
from pyquaternion import Quaternion as PyQuaternion

# ==================== 全局变量定义 ====================
path_yolo = path.join(path.expanduser('~'), 'ur5_project/yolov5-master') # YOLOv5模型路径
path_vision = RosPack().get_path('vision') # vision包路径
path_weigths = path.join(path_vision, 'weigths') # 模型权重文件路径

# 相机和工作台参数
cam_point = (-0.44, -0.5, 1.58)
height_tavolo = 0.74
dist_tavolo = None
origin = None
# 深度学习模型
model = None # YOLO积木检测模型
model_orientation = None  # YOLO积木方向识别模型

legoClasses = ['X1-Y1-Z2', 'X1-Y2-Z1', 'X1-Y2-Z2', 'X1-Y2-Z2-CHAMFER', 'X1-Y2-Z2-TWINFILLET', 'X1-Y3-Z2', 'X1-Y3-Z2-FILLET', 'X1-Y4-Z1', 'X1-Y4-Z2', 'X2-Y2-Z2', 'X2-Y2-Z2-FILLET']
# 命令行参数解析
argv = sys.argv
a_show = '-show' in argv # 是否显示调试图像的标志

# ==================== 工具函数定义 ====================
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

    #color = (120,1,190)
    #mask = get_lego_mask(color, hsv, (5, 5, 5))
    #dist_tavolo = depth[mask].max()
    #if dist_tavolo > 1: dist_tavolo -= height_tavolo
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
    origin = np.array(img.shape[1::-1]) // 2 # [width/2, height/2]

def get_lego_distance(depth):
    """
    获取积木到相机的最近距离
    """
    return depth.min()

def get_lego_color(center, rgb):
    """
    获取指定位置的RGB颜色值
    """
    return rgb[center].tolist()

def get_lego_mask(color, hsv, toll = (20, 20, 255)):
    """
    基于HSV颜色创建二值掩码
    
    参数:
        color: 目标HSV颜色值
        hsv: HSV颜色空间图像
        toll: 颜色容差 (H, S, V)
    
    返回:
        二值掩码图像，目标颜色区域为白色(255)，其他区域为黑色(0)
    
    功能:
        用于分离特定颜色的积木或工作台区域
    """
    thresh = np.array(color)
    mintoll = thresh - np.array([toll[0], toll[1], min(thresh[2]-1, toll[2])])
    maxtoll = thresh + np.array(toll)
    return cv.inRange(hsv, mintoll, maxtoll)

def getDepthAxis(height, lego):
    """
    根据积木高度确定其在深度方向的轴向
    
    参数:
        height: 积木的实际高度 (米)
        lego: 积木类型字符串，如 'X1-Y2-Z2'
    
    返回:
        tuple: (预测尺寸, 轴向索引, 是否匹配)
               轴向索引: 0=X轴, 1=Y轴, 2=Z轴
    
    功能:
        通过分析积木的实际高度，判断积木当前的放置方向
        这对于确定积木的正确姿态至关重要
    """
    X, Y, Z = (int(x) for x in lego[1:8:3])
    #Z = (0.038, 0.057) X = (0.031, 0.063) Y = (0.031, 0.063, 0.095, 0.127)
    rapZ = height / 0.019 - 1
    pinZ = round(rapZ)
    rapXY = height / 0.032
    pinXY = round(rapXY)
    errZ = abs(pinZ - rapZ) + max(pinZ - 2, 0)
    errXY = abs(pinXY - rapXY) + max(pinXY - 4, 0)
    
    if errZ < errXY:
        return pinZ, 2, pinZ == Z    # pin, is ax Z, match
    else:
        if pinXY == Y: return pinXY, 1, True
        else: return pinXY, 0, pinXY == X

def point_distorption(point, height, origin):
    """
    应用透视畸变校正
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
    应用逆透视畸变校正
    参数:
        point: 已畸变的图像坐标点
        height: 物体高度
    
    返回:
        校正前的原始图像坐标
    """
    p = dist_tavolo / (dist_tavolo - height)
    point = point - origin
    return point / p + origin

def myimshow(title, img):
    """
    调试用图像显示函数
    参数:
        title: 窗口标题
        img: 要显示的图像
    功能:
        显示图像并提供鼠标交互功能，点击图像可显示该点的坐标和像素值
        主要用于调试和开发阶段
    """
    def mouseCB(event,x,y,a,b):
        print(x, y, img[y, x], "\r",end='',flush=True)
        print("\033[K", end='')
    cv.imshow(title, img)
    cv.setMouseCallback(title, mouseCB)
    cv.waitKey()

# ==================== 积木定位与识别核心模块 ====================
def process_item(imgs, item):
    """
    处理单个积木的检测、定位和姿态估计
    
    这是整个视觉系统的核心函数，负责:
    1. 从YOLO检测结果中提取积木信息
    2. 进行颜色分析和深度处理
    3. 计算积木的3D位置和姿态
    4. 生成ROS消息格式的结果
    参数:
        imgs: 图像元组 (rgb, hsv, depth, img_draw)
        item: YOLO检测结果字典，包含边界框和分类信息
    返回:
        ModelStates消息对象，包含积木的名称、位置和姿态
        如果处理失败则返回None
    """
    # ========== 第一步：图像数据解包和YOLO结果解析 ==========
    rgb, hsv, depth, img_draw = imgs # 解包图像数据
    # 从YOLO检测结果中提取信息
    x1, y1, x2, y2, cn, cl, nm = item.values() # 边界框坐标、置信度、类别、名称
    # ========== 第二步：边界框处理和安全边距设置 ==========
    mar = 15 # 安全边距，防止边界框超出图像范围
    x1, y1 = max(mar, x1), max(mar, y1)
    x2, y2 = min(rgb.shape[1]-mar, x2), min(rgb.shape[0]-mar, y2)
    boxMin = np.array((x1-mar, y1-mar))
    x1, y1, x2, y2 = np.intp((x1, y1, x2, y2))
    # ========== 第三步：计算积木中心点和颜色信息 ==========
    boxCenter = (y2 + y1) // 2, (x2 + x1) // 2 # 边界框中心点 (y, x)
    color = get_lego_color(boxCenter, rgb) # 获取中心点RGB颜色
    hsvcolor = get_lego_color(boxCenter, hsv) # 获取中心点HSV颜色
    # ========== 第四步：图像区域裁剪 ==========
    sliceBox = slice(y1-mar, y2+mar), slice(x1-mar, x2+mar) # 定义裁剪区域（包含安全边距）

    # 裁剪各类图像到积木区域
    l_rgb = rgb[sliceBox]
    l_hsv = hsv[sliceBox]
    # 如果启用调试模式，在原图上绘制检测框
    if a_show: cv.rectangle(img_draw, (x1,y1),(x2,y2), color, 2)

    l_depth = depth[sliceBox]  # 裁剪深度图像
    # ========== 第五步：深度信息处理和积木分割 ==========
    l_mask = get_lego_mask(hsvcolor, l_hsv) # 基于颜色创建积木掩码
    l_mask = np.where(l_depth < dist_tavolo, l_mask, 0) # 结合深度信息优化掩码（只保留工作台上方的区域）
    l_depth = np.where(l_mask != 0, l_depth, dist_tavolo) # 将非积木区域的深度值设为工作台距离

    # ========== 第六步：积木高度计算 ==========
    #myimshow("asda", hsv)
    # 计算积木到相机的最近距离
    l_dist = get_lego_distance(l_depth)
    l_height = dist_tavolo - l_dist # 计算积木的实际高度
    l_top_mask = cv.inRange(l_depth, l_dist-0.002, l_dist+0.002) # 创建积木顶面掩码（深度值在最近距离附近的区域）
    #cv.bitwise_xor(img_draw,img_draw,img_draw, mask=cv.inRange(depth, l_dist-0.002, l_dist+0.002))
    #myimshow("hmask", l_top_mask)

    # ========== 第七步：积木方向识别 ==========
    depth_borded = np.zeros(depth.shape, dtype=np.float32)
    depth_borded[sliceBox] = l_depth # 将积木区域的深度信息复制到完整图像中
    # 将深度图像标准化为8位灰度图像
    depth_image = cv.normalize(
        depth_borded, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U
    )
    depth_image = cv.cvtColor(depth_image, cv.COLOR_GRAY2RGB).astype(np.uint8) # 转换为3通道RGB图像（YOLO模型需要）
   
    # 使用YOLO方向识别模型
    model_orientation.conf = 0.7 # 设置置信度阈值
    results = model_orientation(depth_image) # 运行方向识别
    # 解析方向识别结果
    pandino = []
    pandino = results.pandas().xyxy[0].to_dict(orient="records")
    n = len(pandino)
    
    # ========== 第八步：积木分类调整和验证 ==========
    pinN, ax, isCorrect = getDepthAxis(l_height, nm) # 根据实际测量的高度调整积木分类
    # 如果分类不正确且当前轴向为Z轴
    if not isCorrect and ax == 2:
        if cl in (1,2,3) and pinN in (1, 2):    # X1-Y2-Z* 系列积木
            cl = 1 if pinN == 1 else 2          # 根据高度调整为对应的Z值
        elif cl in (7, 8) and pinN in (1, 2):   # X1-Y4-Z* 系列积木
            cl = 7 if pinN == 1 else 8          # 根据高度调整为对应的Z值
        elif pinN == -1:
            nm = "{} -> {}".format(nm, "Target") # 标记为目标物体
        else:
            print("[Warning] Error 分类出错！")  # 分类错误警告
    elif not isCorrect: # 如果分类不正确且当前轴向不是Z轴
        ax = 1
        if cl in (0, 2, 5, 8) and pinN <= 4:    # X1-Y*-Z2 系列积木
            cl = (2, 5, 8)[pinN-2]              # 根据尺寸调整Y值
        elif cl in (1, 7) and pinN in (2, 4):   # X1-Y*-Z1 系列积木
            cl = 1 if pinN == 2 else 7          # 根据尺寸调整Y值
        elif cl == 9 and pinN == 1:             # X2-Y2-Z2 特殊情况
            cl = 2                               # 调整为X1类型
            ax = 0                               # 轴向设为X轴
        else: print("[Warning] Error 分类出错！")
    nm = legoClasses[cl] # 更新积木名称

    # ========== 第九步：方向识别结果处理 ==========
    if n != 1: # 如果方向识别结果不唯一
        print("[Warning] 未找到分类")
        or_cn, or_cl, or_nm = ['?']*3 # 设置未知标识
        or_nm = ('lato', 'lato', 'sopra/sotto')[ax]
    else:
        print()
        # 从方向识别结果中提取信息 (class, coordinates, center)
        or_item = pandino[0]
        or_cn, or_cl, or_nm = or_item['confidence'], or_item['class'], or_item['name']
        if or_nm == 'sotto': ax = 2 
        if or_nm == 'lato' and ax == 2: ax = 1
    #---

    # ========== 第十步：积木轮廓检测和几何分析 ==========
    # 检测积木顶面轮廓
    contours, hierarchy = cv.findContours(l_top_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    top_center = np.zeros(2)
    for cnt in contours:
        tmp_center, top_size, top_angle = cv.minAreaRect(cnt)
        top_center += np.array(tmp_center)
    top_center = boxMin + top_center / len(contours)
    # 检测积木整体轮廓
    contours, hierarchy = cv.findContours(l_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return None
    cnt = contours[0]
    l_center, l_size, l_angle = cv.minAreaRect(cnt)
    l_center += boxMin
    if ax != 2: l_angle = top_angle
    l_box = cv.boxPoints((l_center, l_size, l_angle))
    
    # ========== 第十一步：尺寸验证和噪声过滤 ==========
    if l_size[0] <=3 or l_size[1] <=3:
        cv.drawContours(img_draw, np.intp([l_box]), 0, (0,0,0), 2)
        return None # 过滤掉过小的检测结果（可能是噪声）
    
    if a_show: cv.drawContours(img_draw, np.intp([l_box]), 0, color, 2)
    # ========== 第十二步：透视畸变校正 ==========
    # 复制轮廓框用于透视校正
    top_box = l_box.copy()
    # 计算每个顶点到图像中心的距离
    vertexs_norm = [(i, np.linalg.norm(vec - origin)) for vec, i in zip(l_box, range(4))]
    vertexs_norm.sort(key=lambda tup: tup[1])
    # 获取距离图像中心最近的顶点
    iver = vertexs_norm[0][0]
    vec = l_box[iver]
    # 根据积木方向调整高度（如果是顶面朝上，减去一个积木单位高度）
    if or_nm == 'sopra': l_height -= 0.019
    top_box[iver] = point_distorption(l_box[iver], l_height, origin)
    v0 = top_box[iver] - vec
    # 调整相邻顶点的位置以保持矩形形状
    v1 = l_box[iver - 3] - vec    # i - 3 = i+1 % 4
    v2 = l_box[iver - 1] - vec

    top_box[iver - 3] += np.dot(v0, v2) / np.dot(v2, v2) * v2
    top_box[iver - 1] += np.dot(v0, v1) / np.dot(v1, v1) * v1

    l_center = (top_box[0] + top_box[2]) / 2

    if a_show:
        cv.drawContours(img_draw, np.intp([top_box]), 0, (5,5,5), 2)
        cv.circle(img_draw, np.intp(top_box[iver]),1, (0,0,255),1,cv.LINE_AA)    


    # ========== 第十三步：积木坐标系建立和方向向量计算 ==========
    if or_nm in ('sopra', 'sotto', 'sopra/sotto', '?'):  # 确定X,Y方向 (dirX, dirY)
        dirZ = np.array((0,0,1))
        if or_nm == 'sotto': dirZ = np.array((0,0,-1))
        
        projdir = l_center - top_center
        if np.linalg.norm(projdir) < l_size[0] / 10:
            dirY = top_box[0] - top_box[1]
            dirX = top_box[0] - top_box[-1]
            if np.linalg.norm(dirY) < np.linalg.norm(dirX): dirX, dirY = dirY, dirX
            projdir = dirY * np.dot(dirY, projdir)
        edgeFar = [ver for ver in top_box if np.dot(ver - l_center, projdir) >= 0][:2]
        dirY = (edgeFar[0] + edgeFar[1]) / 2 - l_center
        dirY /= np.linalg.norm(dirY)
        dirY = np.array((*dirY, 0))
        dirX = np.cross(dirZ, dirY)

    elif or_nm == "lato": # 侧面朝上时，确定引脚方向 (dirZ)
        edgePin = [ver for ver in top_box if np.dot(ver - l_center, l_center - top_center) >= 0][:2]
        
        dirZ = (edgePin[0] + edgePin[1]) / 2 - l_center
        dirZ /= np.linalg.norm(dirZ)
        dirZ = np.array((*dirZ, 0))
        
        if cl == 10:
            if top_size[1] > top_size[0]: top_size = top_size[::-1]
            if top_size[0] / top_size[1] < 1.7: ax = 0
        if ax == 0:
            vx,vy,x,y = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
            dir = np.array((vx, vy))
            vertexs_distance = [abs(np.dot(ver - l_center, dir)) for ver in edgePin]
            iverFar = np.array(vertexs_distance).argmin()
            
            dirY = edgePin[iverFar] - edgePin[iverFar-1]
            dirY /= np.linalg.norm(dirY)
            dirY = np.array((*dirY, 0))
            dirX = np.cross(dirZ, dirY)
            if a_show: cv.circle(img_draw, np.intp(edgePin[iverFar]), 5, (70,10,50), 1)
            #cv.line(img_draw, np.intp(l_center), np.intp(l_center+np.array([int(vx*100),int(vy*100)])),(0,0,255), 3)
        if ax == 1:
            dirY = np.array((0,0,1))
            dirX = np.cross(dirZ, dirY)

        if a_show: cv.line(img_draw, *np.intp(edgePin), (255,255,0), 2)

    # ========== 第十四步：透视畸变逆校正 ==========
    l_center = point_inverse_distortption(l_center, l_height)

    # ========== 第十五步：积木特定的旋转校正 ==========
    # 这些角度是通过实验标定得到的
    theta = 0
    if cl == 1 and ax == 1: theta = 1.715224 - np.pi / 2
    if cl == 3 and or_nm == 'sotto': theta = 2.359515 - np.pi
    if cl == 4 and ax == 1: theta = 2.145295 - np.pi
    if cl == 6 and or_nm == 'sotto': theta = 2.645291 - np.pi
    if cl == 10 and or_nm == 'sotto': theta = 2.496793 - np.pi

    rotX = PyQuaternion(axis=dirX, angle=theta)
    dirY = rotX.rotate(dirY)
    dirZ = rotX.rotate(dirZ)

    if a_show:
        # 绘制坐标轴框架
        lenFrame = 50
        unit_z = 0.031
        unit_x = 22 * 0.8039 / dist_tavolo
        x_to_z = lenFrame * unit_z/unit_x
        center = np.intp(l_center)

        origin_from_top = origin - l_center
     
        endX = point_distorption(lenFrame * dirX[:2], x_to_z * dirX[2], origin_from_top)
        frameX = (center, center + np.intp(endX))

        endY = point_distorption(lenFrame * dirY[:2], x_to_z * dirY[2], origin_from_top)
        frameY = (center, center + np.intp(endY))
        
        endZ = point_distorption(lenFrame * dirZ[:2], x_to_z * dirZ[2], origin_from_top)
        frameZ = (center, center + np.intp(endZ))
        
        cv.line(img_draw, *frameX, (0,0,255), 2)
        cv.line(img_draw, *frameY, (0,255,0), 2)
        cv.line(img_draw, *frameZ, (255,0,0), 2)
        # ---

        # 绘制文本标注
        if or_cl != '?': or_cn = ['SIDE', 'UP', 'DOWN'][or_cl]
        text = "{} {:.2f} {}".format(nm, cn, or_cn)
        (text_width, text_height) = cv.getTextSize(text, cv.FONT_HERSHEY_DUPLEX, 0.4, 1)[0]
        text_offset_x = boxCenter[1] - text_width // 2
        text_offset_y = y1 - text_height
        box_coords = ((text_offset_x - 1, text_offset_y + 1), (text_offset_x + text_width + 1, text_offset_y - text_height - 1))
        cv.rectangle(img_draw, box_coords[0], box_coords[1], (210,210,10), cv.FILLED)
        cv.putText(img_draw, text, (text_offset_x, text_offset_y), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

    # ========== 第十七步：3D坐标变换和姿态计算 ==========
    def getAngle(vec, ax):
        """计算向量与轴向的夹角"""
        vec = np.array(vec)
        if not vec.any(): return 0
        vec = vec / np.linalg.norm(vec)
        wise = 1 if vec[-1] >= 0 else -1
        dotclamp = max(-1, min(1, np.dot(vec, np.array(ax))))
        return wise * np.arccos(dotclamp)

    msg = ModelStates()
    msg.name = nm
    #fov = 1.047198
    #rap = np.tan(fov)
    #print("rap: ", rap)
    xyz = np.array((l_center[0], l_center[1], l_height / 2 + height_tavolo))
    xyz[:2] /= rgb.shape[1], rgb.shape[0]
    xyz[:2] -= 0.5
    xyz[:2] *= (-0.968, 0.691)
    xyz[:2] *= dist_tavolo / 0.84
    xyz[:2] += cam_point[:2]

    rdirX, rdirY, rdirZ = dirX, dirY, dirZ
    rdirX[0] *= -1
    rdirY[0] *= -1
    rdirZ[0] *= -1 
    qz1 = PyQuaternion(axis=(0,0,1), angle=-getAngle(dirZ[:2], (1,0)))
    rdirZ = qz1.rotate(dirZ)
    qy2 = PyQuaternion(axis=(0,1,0), angle=-getAngle((rdirZ[2],rdirZ[0]), (1,0)))
    rdirX = qy2.rotate(qz1.rotate(rdirX))
    qz3 = PyQuaternion(axis=(0,0,1), angle=-getAngle(rdirX[:2], (1,0)))

    rot = qz3 * qy2 * qz1
    rot = rot.inverse
    msg.pose = Pose(Point(*xyz), Quaternion(x=rot.x,y=rot.y,z=rot.z,w=rot.w))
    
    #pub.publish(msg)
    #print(msg)
    return msg

def process_image(rgb, depth):    
    """
    图像处理主函数：处理同步的RGB和深度图像
    
    参数:
        rgb: OpenCV格式的RGB彩色图像
        depth: OpenCV格式的深度图像
    
    功能流程:
    1. 图像预处理和颜色空间转换
    2. 工作台距离和图像中心计算
    3. YOLO模型目标检测
    4. 逐个处理检测到的积木
    5. 发布ROS消息
    """
    # ========== 图像预处理 ==========
    img_draw = rgb.copy()
    hsv = cv.cvtColor(rgb, cv.COLOR_BGR2HSV) # 转换为HSV颜色空间，便于颜色分割
    # ========== 工作台参数计算 ==========
    get_dist_tavolo(depth, hsv, img_draw)
    get_origin(rgb)

    # ========== YOLO目标检测 ==========
    model.conf = 0.6 # 设置检测置信度阈值
    results = model(rgb)
    pandino = results.pandas().xyxy[0].to_dict(orient="records")
    #print("Model localization: Finish  ")
        
    # ========== 积木详细处理 ==========
    if depth is not None:
        imgs = (rgb, hsv, depth, img_draw)
        # 对每个检测到的积木进行详细处理（3D定位、姿态估计等）
        results = [process_item(imgs, item) for item in pandino]
    
    # ----
    # ========== ROS消息发布 ==========
    msg = ModelStates()
    for point in results:
        if point is not None:
            msg.name.append(point.name)
            msg.pose.append(point.pose)
    pub.publish(msg)

    # ========== 调试显示 ==========
    if a_show:
        cv.imshow("vision-results.png", img_draw)
        cv.waitKey()
    pass

def process_CB(image_rgb, image_depth):
    """
    ROS回调函数：处理接收到的图像消息
    
    参数:
        image_rgb: ROS RGB图像消息
        image_depth: ROS深度图像消息
    
    功能:
    1. 将ROS图像消息转换为OpenCV格式
    2. 调用图像处理函数
    3. 记录处理时间
    4. 关闭ROS节点
    """
    t_start = time.time()
    # ========== ROS图像消息转换 ==========
    # 将ROS RGB图像消息转换为OpenCV BGR格式
    rgb = CvBridge().imgmsg_to_cv2(image_rgb, "bgr8")                                                
    depth = CvBridge().imgmsg_to_cv2(image_depth, "32FC1")
    # ========== 调用图像处理函数 ==========
    process_image(rgb, depth)

    # ========== 性能统计 ==========
    print("处理时间:", time.time() - t_start, "秒")
    
    # 处理完成后关闭ROS节点（单次处理模式）
    rospy.signal_shutdown(0)
    pass

def start_node():
    """
    ROS节点初始化函数：启动视觉识别节点
    
    功能:
    1. 初始化ROS节点
    2. 创建图像订阅器
    3. 创建结果发布器
    4. 设置图像同步机制
    5. 启动消息循环
    """
    global pub

    print("启动视觉识别节点 Vision 1.0")

    rospy.init_node('vision') 
    
    # ========== 创建图像订阅器 ==========
    print("订阅相机图像话题")
    rgb = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth = message_filters.Subscriber("/camera/depth/image_raw", Image)
    
    # ========== 创建结果发布器 ==========
    pub=rospy.Publisher("lego_detections", ModelStates, queue_size=1)

    print("定位系统正在启动...")
    print("(等待图像数据...)", end='\r'), print(end='\033[K')  # 清除行末内容
    
    # ========== 图像同步机制 ==========
    syncro = message_filters.TimeSynchronizer([rgb, depth], 1, reset=True)
    syncro.registerCallback(process_CB)
    
    rospy.spin() 
    pass

def load_models():
    """
    模型加载函数：加载YOLO深度学习模型
    
    功能:
    1. 加载积木分类检测模型
    2. 加载积木方向识别模型
    
    全局变量:
        model: 主要的积木检测和分类模型
        model_orientation: 积木方向识别模型
    """
    global model, model_orientation
    
    # ========== 加载积木分类检测模型 ==========
    print("正在加载积木检测模型 best.pt")
    weight = path.join(path_weigths, 'best.pt')
    model = torch.hub.load(path_yolo,'custom',path=weight, source='local')

    # ========== 加载积木方向识别模型 ==========
    print("正在加载方向识别模型 depth.pt")
    weight = path.join(path_weigths, 'depth.pt')
    model_orientation = torch.hub.load(path_yolo,'custom',path=weight, source='local')
    pass

if __name__ == '__main__':

    load_models()
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
