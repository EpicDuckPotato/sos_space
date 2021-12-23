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
from arm import dynamics, linearize_dynamics, ee_fk, ee_jacobian
from dircol_problem import DircolProblem
from fk_dircol_problem import FKDircolProblem
import crocoddyl as croc

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

# Initialize node and get parameters
rospy.init_node('dircol_node', anonymous=True)
dt = rospy.get_param('dt')
steps = rospy.get_param('planning_steps')
sim_integration = rospy.get_param('sim_integration')
planner_integration = rospy.get_param('planner_integration')
joint_angle_lower_limits = rospy.get_param('joint_angle_lower_limits')
joint_angle_upper_limits = rospy.get_param('joint_angle_upper_limits')
joint_torque_limits = rospy.get_param('joint_torque_limits')
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
goal_pub = rospy.Publisher('/space_robot/goal', PoseStamped, queue_size=1)
goal_pose = PoseStamped()
goal_pose.header.frame_id = 'goal'
goal_pose.pose.position.x = goal_pos[0]
goal_pose.pose.position.y = goal_pos[1]
goal_pose.pose.orientation.z = np.sin(goal_angle/2)
goal_pose.pose.orientation.w = np.cos(goal_angle/2)

plan_pub = rospy.Publisher('/space_robot/plan', Marker, queue_size=1)
plan_marker = Marker()
plan_marker.type = Marker.LINE_STRIP
plan_marker.action = Marker.ADD
plan_marker.pose.position.x = 0
plan_marker.pose.position.y = 0
plan_marker.pose.position.z = 0
plan_marker.pose.orientation.w = 1
plan_marker.pose.orientation.x = 0
plan_marker.pose.orientation.y = 0
plan_marker.pose.orientation.z = 0
plan_marker.scale.x = 0.01
plan_marker.color.r = 1
plan_marker.color.g = 0
plan_marker.color.b = 0
plan_marker.color.a = 1
plan_marker.header.frame_id = "world"
plan_marker.ns = "test_node"

'''
from vm_dynamics import VMDynamics
vmd = VMDynamics()
vmd.euler_lagrange()
'''

arm_dynamics = dynamics(urdf_file)
arm_dynamics_deriv = linearize_dynamics(urdf_file)

Q = 1
R = np.eye(arm_dynamics.num_rotary)
nx = 2*(arm_dynamics.num_rotary + 3)
nu = arm_dynamics.num_rotary
xinit = np.zeros(nx)
xinit[3:3 + arm_dynamics.num_rotary] = np.array(initial_joint_angles)
ulb = -np.array(joint_torque_limits)
uub = np.array(joint_torque_limits)
fk = ee_fk(urdf_file)
jacobian = ee_jacobian(urdf_file)
problem = FKDircolProblem(Q, R, steps, dt, arm_dynamics, arm_dynamics_deriv, nx, nu, xinit, ulb, uub, np.array(goal_pos), fk, jacobian, nx//2, nx//2)
xs, us, solved = problem.solve()
ts = [dt*i for i, x in enumerate(xs)]

robot = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer())
robot.model.gravity.setZero()
model = robot.model

data = pin.Data(model)

for x in xs:
  point = fk(x[:3 + arm_dynamics.num_rotary])
  plan_marker.points.append(Point(point[0], point[1], 0))

t = 0
step = 0
x = np.concatenate((pin.neutral(model), np.zeros(model.nv)))
x[7:model.nq] = np.array(initial_joint_angles)
x[:2] = np.array(initial_base_position)
x[5] = np.sin(initial_base_angle/2)
x[6] = np.cos(initial_base_angle/2)

state = croc.StateMultibody(robot.model)
actuationModel = croc.ActuationModelFloatingBase(state)
action_model = croc.IntegratedActionModelRK4(croc.DifferentialActionModelFreeFwdDynamics(state, actuationModel, croc.CostModelSum(state, len(us[0]))), dt)
action_data = action_model.createData()

rate = rospy.Rate(20)
while not rospy.is_shutdown():
  u = us[step]
  tau = np.concatenate((np.zeros(6), u))
  f_ext = [pin.Force.Zero() for i in range(model.njoints)]
  '''
  q = x[:model.nq]
  v = x[model.nq:]
  acc = pin.aba(model, data, q, v, tau, f_ext)

  vnew = v + acc*dt
  qnew = pin.integrate(model, q, vnew*dt)
  x = np.concatenate((qnew, vnew))
  '''
  action_model.calc(action_data, x, u)
  x = np.copy(action_data.xnext)

  odom = Odometry()
  odom.pose.pose.position.x = x[0]
  odom.pose.pose.position.y = x[1]
  odom.pose.pose.position.z = x[2]
  odom.pose.pose.orientation.x = x[3]
  odom.pose.pose.orientation.y = x[4]
  odom.pose.pose.orientation.z = x[5]
  odom.pose.pose.orientation.w = x[6]
  odom.twist.twist.linear.x = x[model.nq]
  odom.twist.twist.linear.y = x[model.nq + 1]
  odom.twist.twist.linear.z = x[model.nq + 2]
  odom.twist.twist.angular.x = x[model.nq + 3]
  odom.twist.twist.angular.y = x[model.nq + 4]
  odom.twist.twist.angular.z = x[model.nq + 5]
  odom.header.stamp = rospy.Time.now()

  joints = JointState()
  for j in range(arm_dynamics.num_rotary):
    name = model.names[j + 2]
    joints.name.append(name)
    jidx = model.getJointId(name) - 2
    joints.position.append(x[7 + jidx])
    joints.velocity.append(x[model.nq + 6 + jidx])

  joints.header.stamp = rospy.Time.now()
  odom_pub.publish(odom)
  joints_pub.publish(joints)

  plan_marker.header.stamp = rospy.Time.now()
  plan_pub.publish(plan_marker)

  goal_pose.header.stamp = rospy.Time.now()
  goal_pub.publish(goal_pose)

  t += dt
  step += 1

  if step >= steps:
    break

  rate.sleep()
