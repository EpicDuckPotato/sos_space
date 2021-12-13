#!/usr/bin/env python2.7
from nav_msgs.msg import Odometry 
import rospy
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
import numpy as np

base_msg = TransformStamped()
base_msg.header.frame_id = 'world'
base_msg.child_frame_id = 'base_link'
base_msg.transform.rotation.w = 1
base_msg.transform.rotation.x = 0

vm_base_msg = TransformStamped()
vm_base_msg.header.frame_id = 'world'
vm_base_msg.child_frame_id = 'vm_base_link'
vm_base_msg.transform.rotation.w = 1
vm_base_msg.transform.rotation.x = 0
vm_base_msg.transform.translation.x = 1.5

goal_msg = TransformStamped()
goal_msg.header.frame_id = 'world'
goal_msg.child_frame_id = 'goal'
goal_msg.transform.rotation.w = 1
goal_msg.transform.rotation.x = 0

def odom_callback(msg):
  global frame_msg
  base_msg.transform.translation.x = msg.pose.pose.position.x
  base_msg.transform.translation.y = msg.pose.pose.position.y
  base_msg.transform.translation.z = msg.pose.pose.position.z

  base_msg.transform.rotation.x = msg.pose.pose.orientation.x
  base_msg.transform.rotation.y = msg.pose.pose.orientation.y
  base_msg.transform.rotation.z = msg.pose.pose.orientation.z
  base_msg.transform.rotation.w = msg.pose.pose.orientation.w

def goal_callback(msg):
  global goal_msg
  goal_msg.transform.translation.x = msg.pose.position.x
  goal_msg.transform.translation.y = msg.pose.position.y
  goal_msg.transform.translation.z = msg.pose.position.z

  goal_msg.transform.rotation.x = msg.pose.orientation.x
  goal_msg.transform.rotation.y = msg.pose.orientation.y
  goal_msg.transform.rotation.z = msg.pose.orientation.z
  goal_msg.transform.rotation.w = msg.pose.orientation.w

rospy.init_node('tf_node', anonymous=True)
odom_sub = rospy.Subscriber('/space_robot/odom', Odometry, odom_callback)
target_pose_sub = rospy.Subscriber('/space_robot/goal', PoseStamped, goal_callback)

br = TransformBroadcaster()

rate = rospy.Rate(50)
while not rospy.is_shutdown():
  base_msg.header.stamp = rospy.Time.now()
  goal_msg.header.stamp = rospy.Time.now()
  vm_base_msg.header.stamp = rospy.Time.now()
  br.sendTransform(base_msg)
  br.sendTransform(goal_msg)
  br.sendTransform(vm_base_msg)
  rate.sleep()
