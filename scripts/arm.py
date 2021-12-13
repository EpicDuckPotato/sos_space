import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np

class dynamics(object):
  def __init__(self, urdf_file):
    self.robot = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer())
    self.robot.model.gravity.setZero()
    self.model = self.robot.model
    self.num_rotary = self.robot.model.nv - 6
    self.nq = self.robot.model.nq
    self.nv = self.robot.model.nv
    self.rednq = 3 + self.num_rotary

  def __call__(self, x, u):
    data = pin.Data(self.robot.model)
    
    # Turn 2D reduced state into full state
    q = pin.neutral(self.robot.model)
    v = np.zeros(self.nv)

    # Position
    q[:2] = x[:2] 

    # Orientation
    q[5] = np.sin(x[2]/2) 
    q[6] = np.cos(x[2]/2)

    # Joint angles
    q[7:self.nq] = x[3:self.rednq]

    # Linear velocity
    v[:2] = x[self.rednq:self.rednq + 2] 

    # Angular velocity
    v[2] = x[self.rednq + 2] 

    # Joint velocities
    v[6:] = x[self.rednq + 3:] 

    tau = np.zeros(self.nv)
    tau[-self.num_rotary:] = u
    f_ext = [pin.Force.Zero() for i in range(self.robot.model.njoints)]
    acc = pin.aba(self.robot.model, data, q, v, tau, f_ext)

    # Velocities
    xdot = np.zeros(2*self.rednq)
    xdot[:self.rednq] = x[self.rednq:]

    # Linear acceleration
    xdot[self.rednq:self.rednq + 2] = acc[:2]
    
    # Angular acceleration
    xdot[self.rednq + 2] = acc[5]

    # Joint accelerations
    xdot[self.rednq + 3:] = acc[6:]

    return xdot

class linearize_dynamics(object):
  def __init__(self, urdf_file):
    self.robot = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer())
    self.robot.model.gravity.setZero()
    self.model = self.robot.model
    self.num_rotary = self.robot.model.nv - 6
    self.nq = self.robot.model.nq
    self.nv = self.robot.model.nv
    self.rednq = 3 + self.num_rotary

  def __call__(self, x, u):
    data = pin.Data(self.robot.model)
    
    # Turn 2D reduced state into full state
    q = pin.neutral(self.robot.model)
    v = np.zeros(self.nv)

    # Position
    q[:2] = x[:2] 

    # Orientation
    q[5] = np.sin(x[2]/2) 
    q[6] = np.cos(x[2]/2)

    # Joint angles
    q[7:self.nq] = x[3:self.rednq]

    # Linear velocity
    v[:2] = x[self.rednq:self.rednq + 2] 

    # Angular velocity
    v[2] = x[self.rednq + 2] 

    # Joint velocities
    v[6:] = x[self.rednq + 3:] 

    tau = np.zeros(self.nv)
    tau[-self.num_rotary:] = u
    f_ext = [pin.Force.Zero() for i in range(self.robot.model.njoints)]
    pin.computeABADerivatives(self.robot.model, data, q, v, tau, f_ext)

    A = np.zeros((2*self.model.nv, 2*self.model.nv))
    A[:self.model.nv, self.model.nv:] = np.eye(self.model.nv)
    A[self.model.nv:, :self.model.nv] = data.ddq_dq
    A[self.model.nv:, self.model.nv:] = data.ddq_dv

    dtau_du = np.zeros((self.model.nv, self.num_rotary))
    dtau_du[6:] = np.eye(self.num_rotary)
    B = np.zeros((2*self.model.nv, self.num_rotary))
    B[self.model.nv:, :] = np.matmul(data.Minv, dtau_du)

    Ared = self.reduce_matrix(A)
    Bred = np.delete(B, [2, 3, 4, self.nv + 2, self.nv + 3, self.nv + 4], 0)
    return Ared, Bred

  def reduce_matrix(self, m):
    shape = m.shape
    for i in range(len(shape)):
      m = np.delete(m, [2, 3, 4, self.nv + 2, self.nv + 3, self.nv + 4], i)
    return m

class ee_fk(object):
  def __init__(self, urdf_file):
    self.robot = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer())
    self.robot.model.gravity.setZero()
    self.num_rotary = self.robot.model.nv - 6
    self.nq = self.robot.model.nq
    self.nv = self.robot.model.nv
    self.rednq = 3 + self.num_rotary
    self.tip_id = self.robot.model.getFrameId('ee_tip')

  def __call__(self, q):
    data = pin.Data(self.robot.model)
    qfull = pin.neutral(self.robot.model)
    qfull[:2] = q[:2]
    qfull[5] = np.sin(q[2]/2)
    qfull[6] = np.cos(q[2]/2)
    qfull[7:] = q[3:]
    pin.forwardKinematics(self.robot.model, data, qfull)
    pin.updateFramePlacement(self.robot.model, data, self.tip_id)
    return np.copy(data.oMf[self.tip_id].translation[:2])

class ee_jacobian(object):
  def __init__(self, urdf_file):
    self.robot = RobotWrapper.BuildFromURDF(urdf_file, root_joint=pin.JointModelFreeFlyer())
    self.robot.model.gravity.setZero()
    self.num_rotary = self.robot.model.nv - 6
    self.nq = self.robot.model.nq
    self.nv = self.robot.model.nv
    self.rednq = 3 + self.num_rotary
    self.tip_id = self.robot.model.getFrameId('ee_tip')

  def __call__(self, q):
    data = pin.Data(self.robot.model)
    qfull = pin.neutral(self.robot.model)
    qfull[:2] = q[:2]
    qfull[5] = np.sin(q[2]/2)
    qfull[6] = np.cos(q[2]/2)
    qfull[7:] = q[3:]
    J = pin.computeFrameJacobian(self.robot.model, data, qfull, self.tip_id, pin.LOCAL_WORLD_ALIGNED)[:2]
    J = np.delete(J, [2, 3, 4, self.nv + 2, self.nv + 3, self.nv + 4], 1)
    return np.copy(J)
