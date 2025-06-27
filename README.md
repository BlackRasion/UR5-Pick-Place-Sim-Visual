# UR5机械臂视觉仿真系统
> ​​基于ROS/Gazebo的自动化码垛解决方案​​

### 🚀 项目概览
在Ubuntu 20.04环境中，利用ROS Noetic和Gazebo实现UR5机械臂的自动化码垛任务。通过Kinect相机识别工作台上的物块（正方体box1_model和长方体box2_model），控制机械臂将所有物块堆叠成紧凑结构，最小化投影面积并优化仿真时间。

### ⭐ 核心功能
| 模块 | 关键技术 | 功能亮点 |
| ---- | ---- | ---- |
| ​**​世界管理​**​ | 动态场景生成 | 五级难度系统，支持自定义物块布局 |
| ​**​视觉识别​**​ | YOLOv5+传统CV | 混合方案精确识别11种积木+自定义物块 |
| ​**​运动规划​**​ | 逆运动学求解 | 平滑轨迹规划与智能堆叠策略 |
---
### ⚡ 快速开始
**​初始化环境​**​
```
roslaunch levelManager lego_world.launch
```
​**​生成码垛场景​**​
```
rosrun levelManager levelManager.py -1 5
```
​**​启动运动核心系统​**
```
rosrun motion_planning motion_planning.py -l 5 
```
​**​启动视觉核心系统​**
```
rosrun vision box-vision.py -show 
```
---
### 📊 性能指标
**​​堆叠密度​​：**0.000625m²投影面积
**​​效率​​：**101秒完成码垛
**​​稳定性​​：**100%识别率与零坍塌
---
### 🤝 致谢
Thanks to [UR5-Pick-and-Place-Simulation](https://github.com/pietrolechthaler/UR5-Pick-and-Place-Simulation)
