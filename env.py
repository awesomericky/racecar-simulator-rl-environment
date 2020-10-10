# -*- coding: utf-8 -*-
from __future__ import print_function

from copy import deepcopy
import numpy as np
import time
import sys

from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import rospy
import tf

class SpaceSpec:
  def __init__(self, dim, low=None,high=None):
    self.shape = (dim,)
    if low == None:
      self.low = -np.ones(dim, dtype=np.float32)
    else:
      self.low = np.array(low, dtype=np.float32)
    if high == None:
      self.high = np.ones(dim, dtype=np.float32)
    else:
      self.high = np.array(high, dtype=np.float32)

class Env_sim:
  def __init__(self, present_state=True):
    #ros
    rospy.init_node('get_state', anonymous=True)
    self.ros_freq = 25
    self.rate = rospy.Rate(self.ros_freq) #unit : Hz
    self.drive_msg = AckermannDriveStamped()
    self.pose_msg = PoseWithCovarianceStamped()

    #environment
    self.angle_range = np.arange(180, 940, 40) # -90 degree to 90 degree per 10 degrees
    self.limit_distance = 0.2
    self.h_size = 0.5
    self.h_coeff = 10.0
    self.max_step = 1000

    #initial value
    self.sensor_value = np.zeros_like(self.angle_range, dtype=np.float32)
    self.speed = 0.0
    self.hall_sensor_value = 0
    self.pre_hall_sensor_value = 0
    self.cur_step = 0
    self.last_time = rospy.Time.now().to_sec()

    #state & action dimension
    self.action_dim = 2
    self.state_dim = len(self.sensor_value) + self.action_dim
    self.action_space = SpaceSpec(self.action_dim, [-1, -1], [1, 1])
    self.observation_space = SpaceSpec(self.state_dim)
    
    if present_state:
    	self.set_ps()
    	
  def set_ps(self):
    #publisher and subsrciber
    self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)
    self.pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
    self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.state_callback)
    self.hall_sensor_sub = rospy.Subscriber('/odom', Odometry, self.hall_sensor_callback)

  def state_callback(self, data):
    for i, idx in enumerate(self.angle_range):
      self.sensor_value[i] = np.clip(np.mean(data.ranges[idx-40:idx+40]), data.range_min, data.range_max)

  def hall_sensor_callback(self, data):
    lin_vel = data.twist.twist.linear.x/self.ros_freq
    self.hall_sensor_value += lin_vel

  def reset(self):
    time.sleep(1)
    self.rate.sleep()

    #initialize pose
    sample_initial_pos = np.concatenate([np.random.uniform(-5.0, 5.0, 2), [0.0]])
    sample_initial_orn = tf.transformations.quaternion_from_euler(0.0, 0.0, np.random.uniform(-np.pi, np.pi, 1))
    self.pub_pose_data(sample_initial_pos, sample_initial_orn)
    self.pub_drive_data(0.0, 0.0)
    self.rate.sleep()

    #reset value
    self.hall_sensor_value = 0
    self.pre_hall_sensor_value = 0
    self.steering = 0.0
    self.speed = 0
    self.cur_step = 0
    self.stop_cnt = 0

    state = self.get_state()
    self.last_time = rospy.Time.now().to_sec()
    self.info = self.get_info()
    return state

  def step(self, action):
    assert len(action) == self.action_dim
    self.cur_step += 1

    if action[1] < 0.0:
        self.stop_cnt += 1
    else:
        self.stop_cnt = 0

    steering = np.clip(action[0], -1, 1)
    vel = np.clip(action[1], -1, 1)
    vel = 0.0 if vel < 0.0 else 0.5

    if self.cur_step % 10 == 0 or self.cur_step % 10 == 1:
        vel = 0.0

    self.pub_drive_data(vel, steering)
    self.rate.sleep()

    state = self.get_state()
    reward = self.speed

    done = False
    over = False
    if np.min(self.sensor_value) < self.limit_distance:
      done = True
    if self.stop_cnt > 50: #200:
      done = True
    if self.cur_step >= self.max_step:
      done = True
      over = True

    info = self.get_info()
    info['over'] = over
    self.info = info

    if done:
      self.backdrive()

    return state, reward, done, info

  def get_state(self):
    curr_time = rospy.Time.now().to_sec()
    if curr_time - self.last_time >= 0.2:
        self.speed = 0.1*(int(self.hall_sensor_value) - int(self.pre_hall_sensor_value))/(curr_time - self.last_time + 1e-10)
        self.pre_hall_sensor_value = self.hall_sensor_value
        self.last_time = curr_time
    state = np.concatenate([self.sensor_value, [self.steering, self.speed]])
    return state
    
  def get_info(self):
    info = dict()
    dist = np.min(self.sensor_value)
    cost = 1/(1 + np.exp((dist - self.h_size)*self.h_coeff))
    info['inv_sig_cost'] = self.h_size - dist
    info['continuous_cost'] = cost
    info['cost'] = 1.0 if cost > 0.5 else 0.0
    return info

  def backdrive(self):
    self.pub_drive_data(0.0, 0.0)
    self.rate.sleep()

    for i in range(10):
      self.pub_drive_data(-0.75, 0.0)
      self.rate.sleep()

    self.pub_drive_data(0.0, 0.0)
    self.rate.sleep()

  def pub_drive_data(self, speed, steering):
    self.drive_msg.drive.speed = speed*2.0
    self.drive_msg.drive.steering_angle = steering
    self.drive_pub.publish(self.drive_msg)

  def pub_pose_data(self, pos, orn):
    self.pose_msg.pose.pose.position.x = pos[0]
    self.pose_msg.pose.pose.position.y = pos[1]
    self.pose_msg.pose.pose.position.z = pos[2]
    self.pose_msg.pose.pose.orientation.x = orn[0]
    self.pose_msg.pose.pose.orientation.y = orn[1]
    self.pose_msg.pose.pose.orientation.z = orn[2]
    self.pose_msg.pose.pose.orientation.w = orn[3]
    self.pose_msg.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
    self.pose_pub.publish(self.pose_msg)


if __name__ == '__main__':
  env = Env()

  steering = 0
  vel = 0

  for i in range(10):
    print('{} episode start!'.format(i+1))
    s_t = env.reset()
    cnt = 0
    total_cost = 0
    while True:
      if rospy.is_shutdown():
        sys.exit()
      cnt += 1

      action = [0.0, 1.0]
      s_t, r_t, done, info = env.step(action)
      total_cost += info['cost']    
      if done:
        break

    print(total_cost)

