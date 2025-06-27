#!/usr/bin/python3
"""
- 积木与物块场景管理器
  功能：  
- 定义不同类型的物块积木及其属性
- 在指定区域内随机生成物块木
- 支持不同level级别的布局
- 提供物块积木颜色随机化功能
  流程：  
  [选择物块积木类型] --> [计算有效位置] -->[生成随机颜色] --> /  
  [调用Gazebo生成模型] --> [记录物块积木信息]
"""

from pandas import array
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import *
from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_from_euler
import rospy, rospkg, rosservice
import sys
import time
import random
import numpy as np

import xml.etree.ElementTree as ET

# 获取当前包路径
path = rospkg.RosPack().get_path("levelManager")

# 预定义的积木结构类型
costruzioni = ['costruzione-1', 'costruzione-2']

def randomCostruzione():
	"""随机选择一种积木结构类型"""
	return random.choice(costruzioni)

def getPose(modelEl):
	"""
    从XML元素中解析位姿信息
	参数:
        modelEl: XML元素节点
    返回:
        list: [x, y, z, roll, pitch, yaw]
	"""
	strpose = modelEl.find('pose').text
	return [float(x) for x in strpose.split(" ")]

def get_Name_Type(modelEl):
	"""
    从XML元素中获取模型名称和类型
    参数:
        modelEl: XML元素节点
    返回:
        tuple: (完整名称, 基础类型)
    """
	if modelEl.tag == 'model':
		name = modelEl.attrib['name']
	else:
		name = modelEl.find('name').text
	return name, name.split('_')[0]
	
def get_Parent_Child(jointEl):
	"""
    从关节元素中获取父模型和子模型名称
    参数:
        jointEl: 关节XML元素
    返回:
        tuple: (父模型名称, 子模型名称)
    """
	parent = jointEl.find('parent').text.split('::')[0]
	child = jointEl.find('child').text.split('::')[0]
	return parent, child

def getLego4Costruzione(select=None):
	"""
    加载预定义的积木结构
    参数:
        select: 可选，指定加载的结构索引
    返回:
        ModelStates: 包含所有积木位姿的消息
    """
	nome_cost = randomCostruzione()
	if select is not None: nome_cost = costruzioni[select]
	print("spawning", nome_cost)

	tree = ET.parse(f'{path}/lego_models/{nome_cost}/model.sdf')
	root = tree.getroot()
	costruzioneEl = root.find('model')

	brickEls = []
	for modEl in costruzioneEl:
		if modEl.tag in ['model', 'include']:
			brickEls.append(modEl)

	models = ModelStates()
	for bEl in brickEls:
		pose = getPose(bEl)
		models.name.append(get_Name_Type(bEl)[1])
		rot = Quaternion(*quaternion_from_euler(*pose[3:]))
		models.pose.append(Pose(Point(*pose[:3]), rot))

	rospy.init_node("levelManager")
	istruzioni = rospy.Publisher("costruzioneIstruzioni", ModelStates, queue_size=1)
	istruzioni.publish(models)

	return models

def changeModelColor(model_xml, color):
	"""
    修改模型XML中的颜色定义
    参数:
        model_xml: 模型XML字符串
        color: 目标颜色名称
    返回:
        str: 修改后的XML字符串
    """
	root = ET.XML(model_xml)
	root.find('.//material/script/name').text = color
	return ET.tostring(root, encoding='unicode')	


# 默认参数
package_name = "levelManager" # ROS包名
spawn_name = '_spawn' # 生成区域名称
level = None # 当前难度级别
selectBrick = None # 选定的积木类型
maxLego = 11 # 最大积木类型数量
spawn_pos = (-0.35, -0.42, 0.74)  		# 生成区域中心坐标(x,y,z)
spawn_dim = (0.32, 0.23)    			# 生成区域尺寸(长,宽)
min_space = 0.010    					# 积木间最小间距
min_distance = 0.15   					# 积木间最小距离

def readArgs():
	"""
    解析命令行参数
    """
	global package_name
	global level
	global selectBrick
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
				elif arg in ['-b', '-brick']:
					argn += 1
					selectBrick = brickList[int(sys.argv[argn])]
				else:
					raise Exception()
			else: raise Exception()
			argn += 1
	except Exception as err:
		print("Usage: .\levelManager.py" \
					+ "\n\t -l | -level: assigment from 1 to 5" \
					+ "\n\t -b | -brick: spawn specific grip from 0 to 10")
		exit()
		pass

# 积木字典，定义11种不同类型的乐高积木及其尺寸 + 2
brickDict = { \
		'X1-Y1-Z2': (0,(0.031,0.031,0.057)), \
		'X1-Y2-Z1': (1,(0.031,0.063,0.038)), \
		'X1-Y2-Z2': (2,(0.031,0.063,0.057)), \
		'X1-Y2-Z2-CHAMFER': (3,(0.031,0.063,0.057)), \
		'X1-Y2-Z2-TWINFILLET': (4,(0.031,0.063,0.057)), \
		'X1-Y3-Z2': (5,(0.031,0.095,0.057)), \
		'X1-Y3-Z2-FILLET': (6,(0.031,0.095,0.057)), \
		'X1-Y4-Z1': (7,(0.031,0.127,0.038)), \
		'X1-Y4-Z2': (8,(0.031,0.127,0.057)), \
		'X2-Y2-Z2': (9,(0.063,0.063,0.057)), \
		'X2-Y2-Z2-FILLET': (10,(0.063,0.063,0.057)), \
		'box1_model': (11,(0.025,0.025,0.02)), \
		'box2_model': (12,(0.025,0.01,0.02))
		}

brickOrientations = { \
		'X1-Y2-Z1': (((1,1),(1,3)),-1.715224,0.031098), \
		'X1-Y2-Z2-CHAMFER': (((1,1),(1,2),(0,2)),2.359515,0.015460), \
		'X1-Y2-Z2-TWINFILLET': (((1,1),(1,3)),2.145295,0.024437), \
		'X1-Y3-Z2-FILLET': (((1,1),(1,2),(0,2)),2.645291,0.014227), \
		'X1-Y4-Z1': (((1,1),(1,3)),3.14,0.019), \
		'X2-Y2-Z2-FILLET': (((1,1),(1,2),(0,2)),2.496793,0.018718), \
		'box1_model': (((1,1)),0,0.01), \
		'box2_model': (((1,1)),0,0.01)
		} #brickOrientations = (((side, roll), ...), rotX, height)

# 积木颜色列表
colorList = ['Gazebo/Indigo', 'Gazebo/Gray', 'Gazebo/Orange', \
		'Gazebo/Red', 'Gazebo/Purple', 'Gazebo/SkyBlue', \
		'Gazebo/DarkYellow', 'Gazebo/White', 'Gazebo/Green']

brickList = list(brickDict.keys())
counters = [0 for brick in brickList]

# 当前场景中的积木列表
lego = [] 	#lego = [[name, type, pose, radius], ...]

def getModelPath(model):
	'''
	获取模型文件路径
	参数:
		model: 模型名称
	返回:
		str: 模型文件路径
	'''
	pkgPath = rospkg.RosPack().get_path(package_name)
	return f'{pkgPath}/lego_models/{model}/model.sdf'

def randomPose(brickType, rotated):
	"""
    为积木生成随机位姿
    参数:
        brickType: 积木类型
        rotated: 是否允许旋转
    返回:
        tuple: (位姿Pose, 尺寸1, 尺寸2)
    """
	_, dim, = brickDict[brickType]
	spawnX = spawn_dim[0]
	spawnY = spawn_dim[1]
	rotX = 0
	rotY = 0
	rotZ = random.uniform(-3.14, 3.14)
	pointX = random.uniform(-spawnX, spawnX)
	pointY = random.uniform(-spawnY, spawnY)
	pointZ = dim[2]/2
	dim1 = dim[0]
	dim2 = dim[1]
	if rotated:
		side = random.randint(0, 1) 	#0=z/x, 1=z/y
		if (brickType == "X2-Y2-Z2"):
			roll = random.randint(1, 1)	#0=z, 1=x/y, 2=z. 3=x/y
		else:
			roll = random.randint(2, 2) #0=z, 1=x/y, 2=z. 3=x/y

		orients = brickOrientations.get(brickType, ((),0,0) )		
		if (side, roll) not in orients[0]:
			rotX = (side)*roll*1.57
			rotY = (1-side)*roll*1.57
			if roll % 2 != 0:
				dim1 = dim[2]
				dim2 = dim[1-side]
				pointZ = dim[side]/2
		else:
			rotX = orients[1]
			pointZ = orients[2]
			
	rot = Quaternion(*quaternion_from_euler(rotX, rotY, rotZ))
	point = Point(pointX, pointY, pointZ)
	return Pose(point, rot), dim1, dim2

class PoseError(Exception):
	pass

def getValidPose(brickType, rotated):
	"""
    获取不与现有积木碰撞的有效位姿
    参数:
        brickType: 积木类型
        rotated: 是否允许旋转
    返回:
        tuple: (有效位姿, 碰撞半径)
    异常:
        PoseError: 当找不到有效位置时抛出
    """
	trys = 1000
	valid = False
	while not valid:
		pos, dim1, dim2 = randomPose(brickType, rotated)
		radius = np.sqrt((dim1**2 + dim2**2)) / 2
		valid = True
		for brick in lego:
			point = brick[2].position
			r2 = brick[3]
			minDist = max(radius + r2 + min_space, min_distance)
			if (point.x-pos.position.x)**2+(point.y-pos.position.y)**2 < minDist**2:
				valid = False
				trys -= 1
				if trys == 0:
					raise PoseError("Nessun spazio nell'area di spawn")
				break
	return pos, radius

def spawn_model(model, pos, name=None, ref_frame='world', color=None):
	"""
    调用Gazebo服务生成模型
    参数:
        model: 模型类型
        pos: 生成位姿
        name: 可选，模型实例名称
        ref_frame: 参考坐标系
        color: 可选，模型颜色
    返回:
        SpawnModel响应
    """
	if name is None:
		name = model
	
	model_xml = open(getModelPath(model), 'r').read()
	if color is not None:
		model_xml = changeModelColor(model_xml, color)

	spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
	return spawn_model_client(model_name=name, 
	    model_xml=model_xml,
	    robot_namespace='/foo',
	    initial_pose=pos,
	    reference_frame=ref_frame)

def delete_model(name):
	"""
    删除指定名称的模型
    参数:
        name: 模型实例名称
    返回:
        DeleteModel响应
    """
	delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
	return delete_model_client(model_name=name)

def uniformPose(brickType, x, y):
	"""
    为积木生成指定位置的位姿（不旋转）
    参数:
        brickType: 积木类型
        x, y: 指定的x, y坐标
    返回:
        tuple: (位姿Pose, 尺寸1, 尺寸2)
    """
	_, dim, = brickDict[brickType]
	rotX = 0
	rotY = 0
	rotZ = -1.575224   # box-vision +1.575224
	pointX = x
	pointY = y
	pointZ = dim[2]/2
	dim1 = dim[0]
	dim2 = dim[1]
	
	rot = Quaternion(*quaternion_from_euler(rotX, rotY, rotZ))
	point = Point(pointX, pointY, pointZ)
	return Pose(point, rot), dim1, dim2

def spawnaLegoUniform(brickType, x, y):
	"""
    在指定位置生成单个积木（不旋转）
    参数:
        brickType: 积木类型
        x, y: 指定的x, y坐标
    """
	brickIndex = brickDict[brickType][0]
	name = f'{brickType}_{counters[brickIndex]+1}'
	pos, dim1, dim2 = uniformPose(brickType, x, y)
	radius = np.sqrt((dim1**2 + dim2**2)) / 2
	color = random.choice(colorList)

	spawn_model(brickType, pos, name, spawn_name, color)
	lego.append((name, brickType, pos, radius))
	counters[brickIndex] += 1

def generateUniformGrid(brick_types, counts, area_dim):
	"""
    在指定区域内均匀生成多个积木
    参数:
        brick_types: 积木类型列表
        counts: 每种积木的数量列表
        area_dim: 生成区域尺寸 (长, 宽)
    """
	total_blocks = sum(counts)
	if total_blocks == 0:
		return
	
	# 计算网格布局
	cols = int(np.ceil(np.sqrt(total_blocks)))
	rows = int(np.ceil(total_blocks / cols))
	
	# 计算间距
	spacing_x = (2 * area_dim[0]) / (cols + 1)
	spacing_y = (2 * area_dim[1]) / (rows + 1)
	
	# 生成积木位置
	block_index = 0
	for brick_type, count in zip(brick_types, counts):
		for _ in range(count):
			if block_index >= total_blocks:
				break
			
			# 计算网格位置
			row = block_index // cols
			col = block_index % cols
			
			# 计算实际坐标（相对于生成区域中心）
			x = -area_dim[0] + (col + 1) * spacing_x
			y = -area_dim[1] + (row + 1) * spacing_y
			
			# 生成积木
			spawnaLegoUniform(brick_type, x, y)
			block_index += 1

def spawnaLego(brickType=None, rotated=False):
	"""
    生成单个积木的完整流程
    参数:
        brickType: 可选，指定积木类型
        rotated: 是否允许旋转
    """
	if brickType is None:
		brickType = random.choice(brickList)
	
	brickIndex = brickDict[brickType][0]
	name = f'{brickType}_{counters[brickIndex]+1}'
	pos, radius = getValidPose(brickType, rotated)
	color = random.choice(colorList)

	spawn_model(brickType, pos, name, spawn_name, color)
	lego.append((name, brickType, pos, radius))
	counters[brickIndex] += 1

def setUpArea(livello=None, selectBrick=None): 	
	"""
    设置场景的主要函数
    参数:
        livello: 难度级别(1-5)
        selectBrick: 可选，指定积木类型
    """

	# 清理场景中已有的积木
	for brickType in brickList:	#ripulisce
		count = 1
		while delete_model(f'{brickType}_{count}').success: count += 1
	
	# 调用Gazebo服务生成模型
	spawn_model(spawn_name, Pose(Point(*spawn_pos),None) )
	
	try:
		if(livello == 1):
			# 生成单个随机积木
			spawnaLego(selectBrick)
			#spawnaLego('X2-Y2-Z2',rotated=True)
		elif(livello == 2):
			# 所有11个积木
			for brickType in brickList[0:11]:
				spawnaLego(brickType)
		elif(livello == 3):
			# 生成前4个lego积木
			for brickType in brickList[0:4]:
				spawnaLego(brickType)
			# 生成3个旋转的积木
			spawnaLego('X1-Y2-Z2',rotated=True)
			spawnaLego('X1-Y2-Z2',rotated=True)
			spawnaLego('X2-Y2-Z2',rotated=True)
		elif(livello == 4):
			if selectBrick is None:
				# 可搭建的积木组合
				spawn_dim = (0.10, 0.10)    			# 缩小生成区域
				spawnaLego('X1-Y2-Z2',rotated=True)
				spawnaLego('X1-Y2-Z2',rotated=True)
				spawnaLego('X1-Y3-Z2')	
				spawnaLego('X1-Y3-Z2')
				spawnaLego('X1-Y1-Z2')	
				spawnaLego('X1-Y2-Z2-TWINFILLET')
			else:
				models = getLego4Costruzione() # 生成指定组合
				r = 3
				for brickType in models.name:
					r -= 1
					spawnaLego(brickType, rotated=r>0)
		elif(livello == 5):
			spawn_dim = (0.17, 0.17)    			# 缩小生成区域
			# 在缩小区域内均匀生成6个box1_model和3个box2_model
			brick_types = ['box1_model', 'box2_model']
			counts = [6, 3]
			generateUniformGrid(brick_types, counts, spawn_dim)
		else:
			print("[Error]: 选择 1-5")
			return
	except PoseError as err:
		print("[Error]: 生成区域无空间")
		pass
		
	print(f"增添 {len(lego)} 个积木")

if __name__ == '__main__':
	# 解析命令行参数
	readArgs()
	# 检查Gazebo服务是否可用
	try:
		if '/gazebo/spawn_sdf_model' not in rosservice.get_service_list():
			print("等待gazebo服务...")
			rospy.wait_for_service('/gazebo/spawn_sdf_model')
		# 开始场景设置
		setUpArea(level, selectBrick)
		print("场景设置完成，准备开始...")
	except rosservice.ROSServiceIOException as err:
		print("No ROS master execution")
		pass
	except rospy.ROSInterruptException as err:
		print(err)
		pass
	except rospy.service.ServiceException:
		print("No Gazebo services in execution")
		pass
	except rospkg.common.ResourceNotFound:
		print(f"Package not found: '{package_name}'")
		pass
	except FileNotFoundError as err:
		print(f"Model not found: \n{err}")
		print(f"Check model in folder'{package_name}/lego_models'")
		pass
