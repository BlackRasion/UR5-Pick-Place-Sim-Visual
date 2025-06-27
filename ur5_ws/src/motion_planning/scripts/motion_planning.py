#!/usr/bin/python3
"""
UR5机械臂路径规划与运动控制模块

本模块实现了UR5机械臂的路径规划和运动控制功能，主要包括：
1. 乐高积木的检测和识别
2. 机械臂的路径规划和运动控制
3. 夹持器的控制
4. 积木的抓取、调整姿态和放置

输入：通过视觉系统检测到的乐高积木位置和姿态信息
输出：控制机械臂完成积木的抓取和放置任务
"""
import os
import math
import sys
import copy
import json
import actionlib
import control_msgs.msg
from controller import ArmController
from gazebo_msgs.msg import ModelStates
import rospy
from pyquaternion import Quaternion as PyQuaternion
import numpy as np
from gazebo_ros_link_attacher.srv import SetStatic, SetStaticRequest, SetStaticResponse
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse

package_name = "motion_planning" # ROS包名
level = 1 # 当前难度级别

# 获取当前脚本所在目录的路径
PKG_PATH = os.path.dirname(os.path.abspath(__file__))

# 乐高积木模型信息字典
# 包含每种积木类型的目标放置位置（home位置）
# 格式："模型名称": {"home": [x, y, z]}
MODELS_INFO = {
    "X1-Y2-Z1": {
        "home": [0.264589, -0.293903, 0.777] 
    },
    "X2-Y2-Z2": {
        "home": [0.277866, -0.724482, 0.777] 
    },
    "X1-Y3-Z2": {
        "home": [0.268053, -0.513924, 0.777]  
    },
    "X1-Y2-Z2": {
        "home": [0.429198, -0.293903, 0.777] 
    },
    "X1-Y2-Z2-CHAMFER": {
        "home": [0.592619, -0.293903, 0.777]  
    },
    "X1-Y4-Z2": {
        "home": [0.108812, -0.716057, 0.777] 
    },
    "X1-Y1-Z2": {
        "home": [0.088808, -0.295820, 0.777] 
    },
    "X1-Y2-Z2-TWINFILLET": {
        "home": [0.103547, -0.501132, 0.777] 
    },
    "X1-Y3-Z2-FILLET": {
        "home": [0.433739, -0.507130, 0.777]  
    },
    "X1-Y4-Z1": {
        "home": [0.589908, -0.501033, 0.777]  
    },
    "X2-Y2-Z2-FILLET": {
        "home": [0.442505, -0.727271, 0.777] 
    },
    # 新增的自定义积木模型
    "box1_model": {
        "home": [-0.435, -0.505, 0.777]  # 正方体积木的堆叠位置
    },
    "box2_model": {
        "home": [-0.435, -0.505, 0.897]  # 长方体积木的堆叠位置
    }
}
# 可选：调整所有模型的home位置（当前被注释掉）
for model, model_info in MODELS_INFO.items():
    pass
    #MODELS_INFO[model]["home"] = model_info["home"] + np.array([0.0, 0.10, 0.0])

# 从模型文件中读取每个积木的尺寸信息
for model, info in MODELS_INFO.items():
    # 为自定义积木模型手动设置尺寸信息
    if model == "box1_model":
        # 正方体积木尺寸：0.025 x 0.025 x 0.02 (米)
        MODELS_INFO[model]["size"] = (0.025, 0.025, 0.02)
        continue
    elif model == "box2_model":
        # 长方体积木尺寸：0.025 x 0.01 x 0.02 (米)
        MODELS_INFO[model]["size"] = (0.025, 0.01, 0.02)
        continue
    
    model_json_path = os.path.join(PKG_PATH, "..", "models", f"lego_{model}", "model.json")
    # 转换为绝对路径
    model_json_path = os.path.abspath(model_json_path)
    # 检查文件是否存在
    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"模型文件 {model_json_path} 不存在")
    # 读取模型的JSON配置文件
    model_json = json.load(open(model_json_path, "r"))
    corners = np.array(model_json["corners"])

    size_x = (np.max(corners[:, 0]) - np.min(corners[:, 0]))
    size_y = (np.max(corners[:, 1]) - np.min(corners[:, 1]))
    size_z = (np.max(corners[:, 2]) - np.min(corners[:, 2]))

    # print(f"{model}: {size_x:.3f} x {size_y:.3f} x {size_z:.3f}")
    # 将计算得到的尺寸信息存储到MODELS_INFO字典中
    MODELS_INFO[model]["size"] = (size_x, size_y, size_z)

# 乐高积木互锁高度补偿值（米）
INTERLOCKING_OFFSET = 0.019

# 安全位置坐标定义
SAFE_X = -0.40
SAFE_Y = -0.13
SURFACE_Z = 0.774

# 机械臂末端执行器的默认姿态（四元数表示）
DEFAULT_QUAT = PyQuaternion(axis=(0, 1, 0), angle=math.pi)
# 机械臂末端执行器的默认位置
DEFAULT_POS = (-0.1, -0.2, 1.2)
# 默认路径容差设置
DEFAULT_PATH_TOLERANCE = control_msgs.msg.JointTolerance()
DEFAULT_PATH_TOLERANCE.name = "path_tolerance"
DEFAULT_PATH_TOLERANCE.velocity = 10  # 速度容差

def get_gazebo_model_name(model_name, vision_model_pose):
    """
    根据模型名称和视觉检测到的位置，获取Gazebo中对应模型的完整名称
    
    参数:
        model_name: 模型类型名称（如"X1-Y2-Z1"、"box1_model"、"box2_model"）
        vision_model_pose: 视觉系统检测到的模型位姿
    
    返回:
        gazebo_model_name: Gazebo中模型的完整名称
        
    说明:
        这个函数用于链接附着插件，需要精确的Gazebo模型名称
    """
    # 获取Gazebo中所有模型的状态信息
    models = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=None)
    epsilon = 0.05 # 位置匹配的容差范围
    # 遍历所有模型，寻找匹配的模型
    for gazebo_model_name, model_pose in zip(models.name, models.pose):
        # 对于自定义积木模型，直接匹配模型名称
        if model_name in ["box1_model", "box2_model"]:
            if model_name not in gazebo_model_name:
                continue
        # 对于传统乐高积木，使用原有的匹配逻辑
        else:
            if model_name not in gazebo_model_name:
                continue
        # 计算位置差异（曼哈顿距离）
        ds = abs(model_pose.position.x - vision_model_pose.position.x) + abs(model_pose.position.y - vision_model_pose.position.y)
        if ds <= epsilon:
            return gazebo_model_name
     # 如果没有找到匹配的模型，抛出异常
    raise ValueError(f"Model {model_name} at position {vision_model_pose.position.x} {vision_model_pose.position.y} was not found!")


def get_model_name(gazebo_model_name):
    """
    从Gazebo模型的完整名称中提取模型类型名称
    """
    # 处理自定义积木模型（box1_model, box2_model）
    if "box1_model" in gazebo_model_name:
        return "box1_model"
    elif "box2_model" in gazebo_model_name:
        return "box2_model"
    # 处理传统乐高积木模型
    else:
        return gazebo_model_name.replace("lego_", "").split("_", maxsplit=1)[0]


def get_legos_pos(vision=False):
    """
    获取所有乐高积木的位置信息
    
    参数:
        vision: 是否使用视觉检测数据（True）还是直接从Gazebo获取（False）
    
    返回:
        包含(积木名称, 积木位姿)元组的列表
    """
    if vision:
        try:
            if level == 5:
                # level==5时从box_detections话题获取自定义积木信息
                legos = rospy.wait_for_message("/box_detections", ModelStates, timeout=100)
            else:
                # 其他level从lego_detections话题获取传统乐高积木信息
                legos = rospy.wait_for_message("/lego_detections", ModelStates, timeout=100)
        except rospy.ROSException:
            print("视觉检测超时，请检查vision节点")
            return
    else:
        # 从Gazebo模型状态获取积木位置信息
        models = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=None)
        legos = ModelStates()

        # 筛选出乐高积木模型（名称中包含"X"的模型或box模型）
        for name, pose in zip(models.name, models.pose):
            # 检查是否为传统乐高积木（包含"X"）或自定义积木（box1_model, box2_model）
            if "X" not in name and "box" not in name:
                continue
            name = get_model_name(name) # 提取模型类型名称

            legos.name.append(name)
            legos.pose.append(pose)

    return [(lego_name, lego_pose) for lego_name, lego_pose in zip(legos.name, legos.pose)]


def straighten(model_pose, gazebo_model_name):
    """
    调整积木姿态的核心函数
    
    这个函数实现了复杂的路径规划逻辑，根据积木当前的姿态，
    计算合适的抓取方式和调整策略，最终将积木调整为正确的放置姿态。
    
    参数:
        model_pose: 积木当前的位姿信息
        gazebo_model_name: Gazebo中积木的完整名称
    
    功能流程:
        1. 分析积木当前朝向
        2. 计算最佳抓取角度和姿态
        3. 执行抓取动作
        4. 根据需要调整积木姿态
        5. 重新抓取以获得正确的放置姿态
    """
    # 提取积木的位置坐标
    x = model_pose.position.x
    y = model_pose.position.y
    z = model_pose.position.z
    # 将积木的姿态转换为四元数对象
    model_quat = PyQuaternion(
        x=model_pose.orientation.x,
        y=model_pose.orientation.y,
        z=model_pose.orientation.z,
        w=model_pose.orientation.w)
    # 获取积木的尺寸信息
    model_size = MODELS_INFO[get_model_name(gazebo_model_name)]["size"]

    """
    第一步：计算抓取姿态和目标姿态
    """
    # 分析积木当前的朝向（朝上、朝下或侧向）
    facing_direction = get_axis_facing_camera(model_quat)
    # 计算夹持器的最佳接近角度
    approach_angle = get_approach_angle(model_quat, facing_direction)
    # 输出积木朝向信息
    print(f"积木朝向 {facing_direction}")
    # 输出接近角度
    print(f"接近角度 {approach_angle:.2f} deg")

    # 计算抓取时夹持器的姿态
    approach_quat = get_approach_quat(facing_direction, approach_angle)

    # 移动到积木上方，准备抓取
    controller.move_to(x, y, target_quat=approach_quat)

    # 计算目标姿态（用于调整积木方向）
    regrip_quat = DEFAULT_QUAT

    if facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):  # 积木侧向放置
        target_quat = DEFAULT_QUAT
        pitch_angle = -math.pi/2 + 0.2 # 俯仰角度调整
        # 根据接近角度决定旋转方向
        if abs(approach_angle) < math.pi/2:
            target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=math.pi/2)
        else:
            target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi/2)
        target_quat = PyQuaternion(axis=(0, 1, 0), angle=pitch_angle) * target_quat
        # 如果积木朝Y方向，需要额外的旋转调整
        if facing_direction == (0, 1, 0):
            regrip_quat = PyQuaternion(axis=(0, 0, 1), angle=math.pi/2) * regrip_quat

    elif facing_direction == (0, 0, -1):
        """
        倒置积木的预处理步骤
        需要先翻转积木，然后重新定位
        """
        # 下降到积木高度并抓取
        controller.move_to(z=z, target_quat=approach_quat)
        close_gripper(gazebo_model_name, model_size[0])
        # 计算临时姿态（旋转60度）
        tmp_quat = PyQuaternion(axis=(0, 0, 1), angle=2*math.pi/6) * DEFAULT_QUAT
        # 移动到安全位置进行翻转操作
        controller.move_to(SAFE_X, SAFE_Y, z+0.05, target_quat=tmp_quat, z_raise=0.1)  
        controller.move_to(z=z)
        open_gripper(gazebo_model_name)
        # 重新计算抓取姿态
        approach_quat = tmp_quat * PyQuaternion(axis=(1, 0, 0), angle=math.pi/2)
        # 计算最终目标姿态（增加180度偏航旋转）
        target_quat = approach_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi)  
        # 重新抓取时的姿态
        regrip_quat = tmp_quat * PyQuaternion(axis=(0, 0, 1), angle=math.pi)
    else:
        target_quat = DEFAULT_QUAT
        target_quat = target_quat * PyQuaternion(axis=(0, 0, 1), angle=-math.pi/2)

    """
    第二步：执行抓取动作
    根据积木的朝向确定夹持器的闭合程度和抓取高度
    """
    if facing_direction == (0, 0, 1) or facing_direction == (0, 0, -1):
        # 积木正向或倒置时，按X方向尺寸设置夹持器闭合度
        closure = model_size[0]
        z = SURFACE_Z + model_size[2] / 2
    elif facing_direction == (1, 0, 0):
        # 积木朝X方向侧放时，按Y方向尺寸设置夹持器闭合度
        closure = model_size[1]
        z = SURFACE_Z + model_size[0] / 2
    elif facing_direction == (0, 1, 0):
        # 积木朝Y方向侧放时，按X方向尺寸设置夹持器闭合度
        closure = model_size[0]
        z = SURFACE_Z + model_size[1] / 2
     # 移动到计算好的抓取位置并闭合夹持器
    controller.move_to(z=z, target_quat=approach_quat)
    close_gripper(gazebo_model_name, closure)

    """
    第三步：调整积木姿态（如果需要）
    对于非正向放置的积木，需要进行姿态调整
    """
    if facing_direction != (0, 0, 1): # 如果积木不是正向放置
        # 计算调整后的放置高度
        z = SURFACE_Z + model_size[2]/2
        # 抬升积木并调整到目标姿态
        controller.move_to(z=z+0.05, target_quat=target_quat, z_raise=0.1)

        controller.move(dz=-0.05)
        open_gripper(gazebo_model_name)

        # 重新抓取积木（以正确的姿态）
        controller.move_to(z=z, target_quat=regrip_quat, z_raise=0.1)
        close_gripper(gazebo_model_name, model_size[0])

def get_model_link_name(gazebo_model_name):
    """
    根据Gazebo模型名称获取对应的link名称
    
    参数:
        gazebo_model_name: Gazebo中的模型名称
    
    返回:
        str: 对应的link名称
    
    功能:
        不同类型的积木模型在SDF文件中定义的link名称不同：
        - 传统乐高积木：使用 "link" 作为link名称
        - box1_model：使用 "box1" 作为link名称  
        - box2_model：使用 "box2" 作为link名称
    """
    # 根据模型类型返回对应的link名称
    if "box1_model" in gazebo_model_name:
        return "box1"
    elif "box2_model" in gazebo_model_name:
        return "box2"
    else:
        # 传统乐高积木使用 "link" 作为link名称
        return "link"


def close_gripper(gazebo_model_name, closure=0):
    """
    闭合夹持器并建立与物体的动态连接
    
    参数:
        gazebo_model_name: 要抓取的Gazebo模型名称
        closure: 夹持器闭合程度（基于物体尺寸）
    
    功能:
        1. 根据物体尺寸调整夹持器开合度
        2. 通过Gazebo链接附着服务建立物理连接
        3. 自动识别不同模型类型的正确link名称
    """
    # 设置夹持器位置（0.81是最大开合度，根据物体尺寸调整）
    set_gripper(0.81-closure*10)
    # rospy.sleep(0.5)
    rospy.sleep(0.25)
    # 建立动态关节连接（模拟夹持器抓取物体）
    if gazebo_model_name is not None:
        req = AttachRequest()
        req.model_name_1 = gazebo_model_name
        # 根据模型类型获取正确的link名称
        req.link_name_1 = get_model_link_name(gazebo_model_name)
        req.model_name_2 = "robot"
        req.link_name_2 = "wrist_3_link"
        attach_srv.call(req)
        # print(f"已连接模型 {gazebo_model_name}，link名称: {req.link_name_1}")


def open_gripper(gazebo_model_name=None):
    """
    打开夹持器并断开与物体的连接
    
    参数:
        gazebo_model_name: 要释放的Gazebo模型名称
    
    功能:
        1. 完全打开夹持器
        2. 断开与物体的物理连接
        3. 自动识别不同模型类型的正确link名称
    """
    # 完全打开夹持器
    set_gripper(0.0)

    # 断开动态关节连接（释放物体）
    if gazebo_model_name is not None:
        req = AttachRequest()
        req.model_name_1 = gazebo_model_name
        # 根据模型类型获取正确的link名称
        req.link_name_1 = get_model_link_name(gazebo_model_name)
        req.model_name_2 = "robot"
        req.link_name_2 = "wrist_3_link"
        detach_srv.call(req)
        # print(f"已断开模型 {gazebo_model_name}，link名称: {req.link_name_1}")


def set_model_fixed(model_name):
    """
    将模型固定到地面，防止其移动
    
    参数:
        model_name: 要固定的模型名称
    
    功能:
        1. 将模型附着到地面
        2. 设置模型为静态状态
        3. 自动识别不同模型类型的正确link名称
    
    用途:
        在积木放置完成后，防止积木因碰撞而移动
    """
    # 获取模型对应的正确link名称
    link_name = get_model_link_name(model_name)
    
    # 将模型附着到地面
    req = AttachRequest()
    req.model_name_1 = model_name
    req.link_name_1 = link_name
    req.model_name_2 = "ground_plane"
    req.link_name_2 = "link"
    attach_srv.call(req)

    # 设置模型为静态（不受物理影响）
    req = SetStaticRequest()
    # print("{} 放置完毕，link名称: {}".format(model_name, link_name))
    print("{} 放置完毕.".format(model_name))
    req.model_name = model_name
    req.link_name = link_name
    req.set_static = True

    setstatic_srv.call(req)

def get_approach_quat(facing_direction, approach_angle):
    """
    根据积木朝向和接近角度计算夹持器的抓取姿态
    
    参数:
        facing_direction: 积木的朝向向量 (x, y, z)
        approach_angle: 夹持器的接近角度（弧度）
    
    返回:
        PyQuaternion: 夹持器的目标姿态四元数
    
    功能:
        根据积木的不同朝向，计算最优的夹持器接近姿态，
        确保能够稳定可靠地抓取积木
    """
    quat = DEFAULT_QUAT
    if facing_direction == (0, 0, 1):
        pitch_angle = 0
        yaw_angle = 0
    elif facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):
        pitch_angle = + 0.2
        if abs(approach_angle) < math.pi/2:
            yaw_angle = math.pi/2
        else:
            yaw_angle = -math.pi/2
    elif facing_direction == (0, 0, -1):
        pitch_angle = 0
        yaw_angle = 0
    else:
        raise ValueError(f"无效模型状态 {facing_direction}")

    quat = quat * PyQuaternion(axis=(0, 1, 0), angle=pitch_angle)
    quat = quat * PyQuaternion(axis=(0, 0, 1), angle=yaw_angle)
    quat = PyQuaternion(axis=(0, 0, 1), angle=approach_angle+math.pi/2) * quat

    return quat


def get_axis_facing_camera(quat):
    """
    分析积木的朝向，确定其主要面向方向
    
    参数:
        quat: 积木当前的姿态四元数
    
    返回:
        tuple: 朝向向量 (x, y, z)，表示积木的主要朝向
               (0, 0, 1) - 正向朝上
               (0, 0, -1) - 倒置朝下
               (1, 0, 0) - 朝X方向侧放
               (0, 1, 0) - 朝Y方向侧放
    
    功能:
        通过分析积木旋转后的坐标轴方向，判断积木的放置状态，
        为后续的抓取策略提供依据
    """
    axis_x = np.array([1, 0, 0])
    axis_y = np.array([0, 1, 0])
    axis_z = np.array([0, 0, 1])
    new_axis_x = quat.rotate(axis_x)
    new_axis_y = quat.rotate(axis_y)
    new_axis_z = quat.rotate(axis_z)
    # 计算旋转后Z轴与原始Z轴的夹角
    angle = np.arccos(np.clip(np.dot(new_axis_z, axis_z), -1.0, 1.0))
    # 根据角度判断积木的朝向状态
    if angle < np.pi / 3:
        return 0, 0, 1
    elif angle < np.pi / 3 * 2 * 1.2:
        if abs(new_axis_x[2]) > abs(new_axis_y[2]):
            return 1, 0, 0
        else:
            return 0, 1, 0
    else:
        return 0, 0, -1


def get_approach_angle(model_quat, facing_direction):
    """
    计算夹持器接近积木的最佳角度
    
    参数:
        model_quat: 积木当前的姿态四元数
        facing_direction: 积木的朝向向量
    
    返回:
        float: 夹持器的接近角度（弧度）
    
    功能:
        根据积木的当前姿态和朝向，计算夹持器应该以什么角度接近积木，
        以确保最佳的抓取效果
    """
    if facing_direction == (0, 0, 1):
        # 使用积木的偏航角减去90度作为接近角度
        return model_quat.yaw_pitch_roll[0] - math.pi/2 
    elif facing_direction == (1, 0, 0) or facing_direction == (0, 1, 0):
        axis_x = np.array([0, 1, 0])
        axis_y = np.array([-1, 0, 0])
        # 获取积木旋转后的Z轴方向
        new_axis_z = model_quat.rotate(np.array([0, 0, 1]))
        # 计算积木Z轴与参考轴的角度关系
        dot = np.clip(np.dot(new_axis_z, axis_x), -1.0, 1.0) # 正弦值
        det = np.clip(np.dot(new_axis_z, axis_y), -1.0, 1.0) # 余弦值
        return math.atan2(det, dot) # 使用atan2计算角度
    elif facing_direction == (0, 0, -1):
        # 对于倒置积木，使用相反的角度计算
        return -(model_quat.yaw_pitch_roll[0] - math.pi/2) % math.pi - math.pi
    else:
        raise ValueError(f"无效模型状态 {facing_direction}")


def set_gripper(value):
    """
    设置夹持器的开合程度
    
    参数:
        value: 夹持器位置值（0.0完全打开，0.8完全闭合）
    
    返回:
        夹持器动作的执行结果
    
    功能:
        通过ROS Action接口控制夹持器的开合，实现对物体的抓取和释放
    """
    goal = control_msgs.msg.GripperCommandGoal()
    goal.command.position = value  
    goal.command.max_effort = -1  # 不限制力度
    action_gripper.send_goal_and_wait(goal, rospy.Duration(10))

    return action_gripper.get_result()

def process_same_type_stacking(legos):
    """
    处理同类型积木的堆叠功能（仅用于level == 5）
    
    参数:
        legos: 积木列表，包含(积木名称, 积木位姿)元组
    
    功能:
        1. 按积木类型进行分组
        2. 对每种类型的积木进行同类堆叠
        3. 显示处理进度和结果统计
    返回:
        无
    """
    # 按模型类型对积木进行分组，实现同类堆叠
    lego_groups = {}
    for model_name, model_pose in legos:
        if model_name not in lego_groups:
            lego_groups[model_name] = []
        lego_groups[model_name].append((model_name, model_pose))
    
    # 对每组积木按位置排序（按X、Y坐标升序排列）
    for model_type in lego_groups:
        lego_groups[model_type].sort(
            key=lambda a: (a[1].position.x, -a[1].position.y))
    
    print(f"检测到的积木类型: {list(lego_groups.keys())}")
    for model_type, group in lego_groups.items():
        print(f"  {model_type}: {len(group)}个")
    
    # 检查是否有未识别的积木类型
    unknown_types = []
    for model_type in lego_groups.keys():
        if model_type not in MODELS_INFO:
            unknown_types.append(model_type)
    
    if unknown_types:
        print(f"警告：发现未识别的积木类型: {unknown_types}")
        print("这些积木将被跳过处理")
    
    print("\n=== 开始同类堆叠处理 ===")
    z_offet = 0.0  #  lego 堆叠时，z轴的偏移量
    # 按指定顺序处理积木类型：box1_model优先，然后box2_model，最后其他类型
    processing_order = []
    if 'box1_model' in lego_groups:
        processing_order.append('box1_model')
    # 添加其他类型
    for model_type in lego_groups:
        if model_type != 'box1_model':
            processing_order.append(model_type)
    
    # 遍历所有积木组，实现同类堆叠
    for model_type in processing_order:
        group = lego_groups[model_type]
        print(f"\n开始处理 {model_type} 积木，共 {len(group)} 个")
        
        # 重置该类型积木的堆叠高度
        if model_type in MODELS_INFO:
            original_home = MODELS_INFO[model_type]["home"].copy()
        
        # 遍历该类型的所有积木
        for i, (model_name, model_pose) in enumerate(group):
            print(f"\n处理第 {i+1}/{len(group)} 个 {model_type} 积木")
            # 动态设置目标点
            if(model_type == "box1_model" and i == 0):
                MODELS_INFO["box1_model"]["home"][0] = model_pose.position.x
                MODELS_INFO["box2_model"]["home"][0] = model_pose.position.x
                MODELS_INFO["box1_model"]["home"][1] = model_pose.position.y
                MODELS_INFO["box2_model"]["home"][1] = model_pose.position.y

            open_gripper() # 打开夹持器准备抓取
            try:
                # 获取积木的目标位置和尺寸信息
                model_home = MODELS_INFO[model_name]["home"]
                model_size = MODELS_INFO[model_name]["size"]
            except KeyError as e:
                print(f"积木 {model_name} 类型未识别!")  # 积木类型未识别
                continue

            # 获取Gazebo中对应的模型名称
            try:
                gazebo_model_name = get_gazebo_model_name(model_name, model_pose)
            except ValueError as e:
                print(e)
                continue

            # 执行积木姿态调整和抓取
            print(f"积木 {model_name} 在位置 {model_pose.position.x:.3f}, {model_pose.position.y:.3f}, {model_pose.position.z:.3f}") # 输出积木位置
            straighten(model_pose, gazebo_model_name) # 调整积木姿态            
            controller.move(dz=(0.05 + z_offet)) # 抓取后向上移动，避免碰撞
            z_offet += 0.02  #  lego 堆叠时，z轴的偏移量

            # 移动到目标堆叠位置
            x, y, z = model_home
            z += model_size[2] / 2 - 0.00001 # 计算放置高度（积木中心高度）
            print(f"移动积木 {model_name} 到 {x:.3f} {y:.3f} {z:.3f}")
            # 移动到目标位置上方，调整姿态为垂直放置
            controller.move_to(x, y, target_quat=DEFAULT_QUAT * PyQuaternion(axis=[0, 0, 1], angle=math.pi / 2))
            # 下降到目标高度并释放积木
            controller.move_to(x, y, z)
            set_model_fixed(gazebo_model_name)
            open_gripper(gazebo_model_name)
            controller.move(dz=0.06) # 释放后向上移动，避免碰撞
            # 如果机械臂位置超出安全范围，返回默认位置
            if controller.gripper_pose[0][1] > -0.3 and controller.gripper_pose[0][0] > 0:
                controller.move_to(*DEFAULT_POS, DEFAULT_QUAT)

            # 更新堆叠高度，为下一个同类型积木做准备
            MODELS_INFO[model_name]["home"][2] += model_size[2]
            
        print(f"完成 {model_type} 积木的堆叠，共处理 {len(group)} 个")
    
    # 显示处理结果统计
    print("\n=== 所有积木处理完成 ===")
    total_processed = sum(len(group) for group in lego_groups.values())
    print(f"总共处理了 {total_processed} 个积木")
    print(f"成功实现了 {len(lego_groups)} 种类型积木的同类堆叠")
    
    # 显示最终堆叠结果
    print("\n最终堆叠位置:")
    for model_type, group in lego_groups.items():
        if model_type in MODELS_INFO:
            final_height = MODELS_INFO[model_type]["home"][2]
            print(f"  {model_type}: {len(group)}层，最终高度 {final_height:.3f}m")


def process_original_stacking(legos):
    """
    处理原有的积木堆叠逻辑（用于level != 5的情况）
    
    参数:
        legos: 积木列表，包含(积木名称, 积木位姿)元组
    
    功能:
        按原有逻辑处理积木，按位置排序后逐个处理
    
    返回:
        无
    """
    # 按X、Y坐标降序排列积木（原有逻辑）
    legos.sort(reverse=True, key=lambda a: (a[1].position.x, a[1].position.y))
    
    # 遍历所有检测到的积木，逐个进行抓取和堆叠
    for model_name, model_pose in legos:
        open_gripper() # 打开夹持器准备抓取
        try:
            # 获取积木的目标位置和尺寸信息
            model_home = MODELS_INFO[model_name]["home"]
            model_size = MODELS_INFO[model_name]["size"]
        except KeyError as e:
            print(f"积木 {model_name} 类型未识别!")  # 积木类型未识别
            continue

        # 获取Gazebo中对应的模型名称
        try:
            gazebo_model_name = get_gazebo_model_name(model_name, model_pose)
        except ValueError as e:
            print(e)
            continue

        # 执行积木姿态调整和抓取
        print(f"积木 {model_name} 在位置 {model_pose.position.x:.3f}, {model_pose.position.y:.3f}, {model_pose.position.z:.3f}") # 输出积木位置
        straighten(model_pose, gazebo_model_name) # 调整积木姿态
        controller.move(dz=0.15) # 抓取后向上移动，避免碰撞

        # 移动到目标堆叠位置
        x, y, z = model_home
        z += model_size[2] / 2 + 0.004 # 计算放置高度（积木中心高度+微调偏移）
        print(f"移动积木 {model_name} 到 {x:.3f} {y:.3f} {z:.3f}")
        # 移动到目标位置上方，调整姿态为垂直放置
        controller.move_to(x, y, target_quat=DEFAULT_QUAT * PyQuaternion(axis=[0, 0, 1], angle=math.pi / 2))
        # 下降到目标高度并释放积木
        controller.move_to(x, y, z)
        set_model_fixed(gazebo_model_name)
        open_gripper(gazebo_model_name)
        controller.move(dz=0.15) # 释放后向上移动，避免碰撞
        # 如果机械臂位置超出安全范围，返回默认位置
        if controller.gripper_pose[0][1] > -0.3 and controller.gripper_pose[0][0] > 0:
            controller.move_to(*DEFAULT_POS, DEFAULT_QUAT)

        # 更新堆叠高度，为下一个积木做准备
        MODELS_INFO[model_name]["home"][2] += model_size[2] - INTERLOCKING_OFFSET

def readArgs():
	"""
    解析命令行参数
    """
	global package_name
	global level
	try:
		argn = 1
		while argn < len(sys.argv):
			arg = sys.argv[argn]
			
			if arg.startswith('__'):
				None
			elif arg[0] == '-':
				if arg in ['-l', '-level']:
					argn += 1
					level = int(sys.argv[argn])
				else:
					raise Exception()
			else: raise Exception()
			argn += 1
	except Exception as err:
		print("Usage: .\motion_planning.py" \
					+ "\n\t -l | -level: assigment from 1 to 5")
		exit()
		pass

if __name__ == "__main__":
    """
    主程序入口 - UR5机械臂乐高积木抓取与堆叠系统
    
    系统功能:
        1. 初始化ROS节点和各种服务客户端
        2. 获取视觉系统检测到的积木位置信息
        3. 对每个积木执行：抓取 -> 姿态调整 -> 移动到目标位置 -> 释放
        4. 实现积木的自动分类堆叠
    
    工作流程:
        - 机械臂移动到默认位置
        - 按X坐标排序处理积木
        - 对每个积木调用straighten函数进行姿态调整
        - 将积木移动到对应的目标堆叠位置
        - 更新堆叠高度，为下一个积木做准备
    """
    # 解析命令行参数
    readArgs()
    print(f"使用关卡级别: {level}")

    print("kinematics运动学节点初始化..")
    rospy.init_node("send_joints")
    # 初始化机械臂控制器
    controller = ArmController()

    # 创建夹持器动作客户端
    action_gripper = actionlib.SimpleActionClient(
        "/gripper_controller/gripper_cmd",
        control_msgs.msg.GripperCommandAction
    )
    print("等待夹持器控制器服务启动...")
    action_gripper.wait_for_server()
    # 创建Gazebo连接插件的服务代理
    setstatic_srv = rospy.ServiceProxy("/link_attacher_node/setstatic", SetStatic) # 设置模型固定服务
    attach_srv = rospy.ServiceProxy("/link_attacher_node/attach", Attach) # 连接模型服务
    detach_srv = rospy.ServiceProxy("/link_attacher_node/detach", Attach) # 断开模型连接服务
    setstatic_srv.wait_for_service()
    attach_srv.wait_for_service()
    detach_srv.wait_for_service()
    # 机械臂移动到默认位置
    controller.move_to(*DEFAULT_POS, DEFAULT_QUAT)

    print("等待视觉检测的积木位置信息...") 
    # rospy.sleep(0.5)
    rospy.sleep(0.25)
    # legos = get_legos_pos(vision=True) # 获取视觉检测到的积木位置信息
    legos = get_legos_pos(vision=False)
    print(f"检测到 {len(legos)} 个积木")


    
    # 根据关卡选择处理方式
    if level == 5:
        print("使用同类堆叠模式 (level == 5)")
        process_same_type_stacking(legos)
    else:
        print(f"使用原有堆叠模式 (level = {level})")
        process_original_stacking(legos)
    
    print("\n返回默认位置")
    controller.move_to(*DEFAULT_POS, DEFAULT_QUAT)
    open_gripper()
    rospy.sleep(0.4)
