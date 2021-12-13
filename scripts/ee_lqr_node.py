#!/usr/bin/env python3.6

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import rospy
import rospkg
import numpy as np
import time
import threading
from std_msgs.msg import Bool, Float64, ColorRGBA
from nav_msgs.msg import Odometry 
from sensor_msgs.msg import JointState 
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Point, PoseStamped
from scipy.spatial.transform import Rotation as R
from scipy.linalg import solve_continuous_are
from vm_dynamics import VMDynamics

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

# Initialize node and get parameters
rospy.init_node('docking_node', anonymous=True)
dt = rospy.get_param('dt')
steps = rospy.get_param('planning_steps')
sim_integration = rospy.get_param('sim_integration')
planner_integration = rospy.get_param('planner_integration')
joint_angle_lower_limits = rospy.get_param('joint_angle_lower_limits')
joint_angle_upper_limits = rospy.get_param('joint_angle_upper_limits')
joint_torque_limits = rospy.get_param('joint_torque_limits')
constraint_polishing = rospy.get_param('constraint_polishing')
goal_pos = rospy.get_param('goal_position')
goal_angle = rospy.get_param('goal_angle')
initial_joint_angles = rospy.get_param('initial_joint_angles')
initial_base_position = rospy.get_param('initial_base_position')
initial_base_angle = rospy.get_param('initial_base_angle')

# Instantiate planner with urdf file
rospack = rospkg.RosPack()
path = rospack.get_path('sos_space')
urdf_file = path + '/urdf/robot.urdf'

# Publishers
odom_pub = rospy.Publisher('/space_robot/odom', Odometry, queue_size=1)
joints_pub = rospy.Publisher('/space_robot/fbk/joint_state', JointState, queue_size=1)

t = 0
step = 0

robot = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer())
robot.model.gravity.setZero()
model = robot.model
data = pin.Data(model)
x = np.concatenate((pin.neutral(model), np.zeros(model.nv)))
x[7:model.nq] = np.array(initial_joint_angles)
x[:2] = np.array(initial_base_position)
x[5] = np.sin(initial_base_angle/2)
x[6] = np.cos(initial_base_angle/2)
q = x[:model.nq]
v = x[model.nq:]
num_rotary = len(initial_joint_angles)

vmd = VMDynamics()
vmd.euler_lagrange()
q0 = np.array([0, 1, -1, 0.5])
qdot0 = np.zeros(4)
u0 = np.zeros(3)
A, B = vmd.linearize_ee(q0, qdot0, u0)
print(A)
print(B)

Q = np.zeros((12, 12))
#Q[8:12, 8:12] = np.eye(4)
Q[1:4, 1:4] = np.eye(3)
Q[5:8, 5:8] = np.eye(3)
r = np.eye(num_rotary)
S = solve_continuous_are(A, B, Q, r)
K = np.linalg.solve(r, np.matmul(B.transpose(), S))
K = np.concatenate((np.zeros((num_rotary, 5)), K), 1)
K = np.insert(K, 6 + num_rotary, np.zeros((5, num_rotary)), 1)
print(K)

q0 = pin.neutral(model)
v0 = np.zeros(model.nv)

pin.forwardKinematics(model, data, q0, v0)
tip_id = model.getFrameId('ee_tip')
pin.updateFramePlacement(model, data, tip_id)
ee0 = np.copy(data.oMf[tip_id].translation[:2])

rate = rospy.Rate(20)
while not rospy.is_shutdown():
  q = x[:model.nq]
  v = x[model.nq:]
  xdiff = np.concatenate((pin.difference(model, q0, q), v - v0))
  pin.forwardKinematics(model, data, q, v)
  pin.updateFramePlacement(model, data, tip_id)
  ee_pos = data.oMf[tip_id].translation[:2]
  J = pin.computeFrameJacobian(model, data, q, tip_id, pin.LOCAL_WORLD_ALIGNED)
  ee_vel = np.matmul(J, v)
  xdiff = np.concatenate((xdiff, ee_pos - ee0, ee_vel))
  u = -np.matmul(K, xdiff)
  tau = np.concatenate((np.zeros(6), u))
  f_ext = [pin.Force.Zero() for i in range(model.njoints)]
  acc = pin.aba(model, data, q, v, tau, f_ext)
  vnew = v + acc*dt
  qnew = pin.integrate(model, q, vnew*dt)
  x = np.concatenate((qnew, vnew))

  odom = Odometry()
  joints = JointState()

  odom.pose.pose.position.x = x[0]
  odom.pose.pose.position.y = x[1]
  odom.pose.pose.position.z = x[2]

  odom.pose.pose.orientation.x = x[3]
  odom.pose.pose.orientation.y = x[4]
  odom.pose.pose.orientation.z = x[5]
  odom.pose.pose.orientation.w = x[6]

  R_base_world = R.from_quat(x[3:7])
  v = R_base_world.apply(x[model.nq:model.nq + 3])
  w = R_base_world.apply(x[model.nq + 3:model.nq + 6])
  odom.twist.twist.linear.x = v[0]
  odom.twist.twist.linear.y = v[1]
  odom.twist.twist.linear.z = v[2]
  odom.twist.twist.angular.x = w[0]
  odom.twist.twist.angular.y = w[1]
  odom.twist.twist.angular.z = w[2]

  for j in range(num_rotary):
    name = model.names[j + 2]
    joints.name.append(name)
    jidx = model.getJointId(name) - 2
    joints.position.append(x[7 + jidx])
    joints.velocity.append(x[model.nq + 6 + jidx])

  odom.header.stamp = rospy.Time.now()
  joints.header.stamp = rospy.Time.now()
  odom_pub.publish(odom)
  joints_pub.publish(joints)

  t += dt
  step += 1

  rate.sleep()
