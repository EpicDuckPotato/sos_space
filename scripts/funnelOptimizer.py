import pinocchio as pin
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian
from pydrake.symbolic import TaylorExpand, cos, sin
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.linalg import solve_continuous_are

class FunnelOptimizer(object):
  def __init__(self, model, dt, ts, us, xs, goal_pos):
    self.model = model
    self.dt = dt
    self.ts = ts
    self.us = us
    self.xs = xs
    self.num_rotary = self.model.nv - 6

    for i in range(len(self.xs)):
      q = pin.neutral(self.model)
      v = np.zeros(self.model.nv)
      q[:2] = self.xs[i][:2]
      q[5] = np.sin(self.xs[i][2]/2)
      q[6] = np.cos(self.xs[i][2]/2)
      q[7:] = self.xs[i][3:3 + self.num_rotary]
      v[:2] = self.xs[i][3 + self.num_rotary:3 + self.num_rotary + 2]
      v[5] = self.xs[i][3 + self.num_rotary + 2]
      v[6:] = self.xs[i][3 + self.num_rotary + 3:]
      self.xs[i] = np.concatenate((q, v))
      
    self.goal_pos = goal_pos
    self.steps = len(self.xs) - 1

    self.datas = [pin.Data(self.model) for step in range(self.steps + 1)]

    # Get terminal value function using infinite horizon LQR
    K, S = self.terminal_lqr()
    self.us.append(np.zeros_like(self.us[-1]))
    self.K = K
    self.S = S

  def fbk(self, x):
    state_diff = np.zeros(x.shape[0] - 1)
    state_diff[:3] = x[:3] - self.xs[-1][:3]
    state_diff[6:] = x[7:] - self.xs[-1][7:]
    R0 = R.from_quat(self.xs[-1][3:7])
    R1 = R.from_quat(x[3:7])
    Rdiff = R0.inv()*R1
    state_diff[3:6] = Rdiff.as_rotvec()

    return self.us[-1] - np.matmul(self.K, state_diff)

  def terminal_lqr(self):
    data = pin.Data(self.model)
    q = self.xs[-1][:self.model.nq]
    v = self.xs[-1][self.model.nq:]

    pin.forwardKinematics(self.model, data, q, v)
    tip_id = self.model.getFrameId('ee_tip')
    pin.updateFramePlacement(self.model, data, tip_id)
    J = pin.computeFrameJacobian(self.model, data, q, tip_id, pin.LOCAL_WORLD_ALIGNED)[:2]
    Q = np.zeros((2*self.model.nv, 2*self.model.nv))
    Q[:self.model.nv, :self.model.nv] = np.matmul(J.transpose(), J)
    Q[self.model.nv:, self.model.nv:] = np.matmul(J.transpose(), J)

    tau = np.zeros(self.model.nv)
    fext = [pin.Force.Zero() for i in range(self.model.njoints)]
    pin.computeABADerivatives(self.model, data, q, v, tau, fext)
    A = np.zeros((2*self.model.nv, 2*self.model.nv))
    A[:self.model.nv, self.model.nv:] = np.eye(self.model.nv)
    A[self.model.nv:, :self.model.nv] = data.ddq_dq
    A[self.model.nv:, self.model.nv:] = data.ddq_dv

    dtau_du = np.zeros((self.model.nv, self.num_rotary))
    dtau_du[6:] = np.eye(self.num_rotary)
    B = np.zeros((2*self.model.nv, self.num_rotary))
    B[self.model.nv:, :] = np.matmul(data.Minv, dtau_du)

    R = np.eye(self.num_rotary)

    Ared = self.reduce_matrix(A)
    Qred = self.reduce_matrix(Q)
    Bred = np.delete(B, [2, 3, 4, self.model.nv + 2, self.model.nv + 3, self.model.nv + 4], 0)
    S = solve_continuous_are(Ared, Bred, Qred, R)
    S = np.insert(S, 2, np.zeros((3, 2*(3 + self.num_rotary))), 0)
    S = np.insert(S, self.model.nv + 2, np.zeros((3, 2*(3 + self.num_rotary))), 0)
    S = np.insert(S, 2, np.zeros((3, 2*self.model.nv)), 1)
    S = np.insert(S, self.model.nv + 2, np.zeros((3, 2*self.model.nv)), 1)

    K = np.linalg.solve(R, np.matmul(B.transpose(), S))
    return K, S

  def reduce_matrix(self, m):
    shape = m.shape
    for i in range(len(shape)):
      m = np.delete(m, [2, 3, 4, self.model.nv + 2, self.model.nv + 3, self.model.nv + 4], i)
    return m

  # Analytical 1st derivative
  def get_d1(self, x, step):
    q = x[:self.model.nq]
    v = x[self.model.nq:]
    data = self.datas[step]
    u = self.fbk(x)
    tau = np.concatenate((np.zeros(6), u))
    fext = [pin.Force.Zero() for i in range(self.model.njoints)]
    pin.computeABADerivatives(self.model, data, q, v, tau, fext)
    d1 = np.zeros((2*self.model.nv, 2*self.model.nv))
    d1[:self.model.nv, self.model.nv:] = np.eye(self.model.nv)
    d1[self.model.nv:, :self.model.nv] = self.datas[step].ddq_dq
    d1[self.model.nv:, self.model.nv:] = self.datas[step].ddq_dv
    dtau_dx = np.concatenate((np.zeros((6, 2*self.model.nv)), self.K))
    d1[self.model.nv:] -= np.matmul(data.Minv, dtau_dx)
    return d1

  # Finite-differenced 2nd derivative
  def get_d2(self, x, eps, d1, step):
    q = x[:self.model.nq]
    v = x[self.model.nq:]

    d2 = np.zeros((2*self.model.nv, 2*self.model.nv, 2*self.model.nv))

    for i in range(2*self.model.nv):
      perturbation = np.zeros(self.model.nv)
      if i < self.model.nv:
        perturbation[i] = eps
        qnew = pin.integrate(self.model, x[:self.model.nq], perturbation)
        xnew = np.concatenate((qnew, v))
      else:
        perturbation[i - self.model.nv] = eps
        xnew = np.concatenate((q, v + perturbation))

      newd1 = self.get_d1(xnew, step)
      d2[:, :, i] = (newd1 - d1)/eps

    return d2

  # Finite-differenced 3nd derivative
  def get_d3(self, x, eps, d1, d2, step):
    q = x[:self.model.nq]
    v = x[self.model.nq:]

    d3 = np.zeros((2*self.model.nv, 2*self.model.nv, 2*self.model.nv, 2*self.model.nv))

    for i in range(2*self.model.nv):
      perturbation = np.zeros(self.model.nv)
      if i < self.model.nv:
        perturbation[i] = eps
        qnew = pin.integrate(self.model, x[:self.model.nq], perturbation)
        xnew = np.concatenate((qnew, v))
      else:
        perturbation[i - self.model.nv] = eps
        xnew = np.concatenate((q, v + perturbation))

      newd2 = self.get_d2(xnew, eps, d1, step)
      d3[:, :, :, i] = (newd2 - d2)/eps

    return d3

  # Return 3rd order Taylor expansion of dynamics. xerr is state reduced to 2D
  def error_dynamics(self, step, xerr):
    eps = 0.001
    d1 = self.get_d1(self.xs[step], step)
    d2 = self.get_d2(self.xs[step], eps, d1, step)
    d3 = self.get_d3(self.xs[step], eps, d1, d2, step)

    d1 = self.reduce_matrix(d1)
    d2 = self.reduce_matrix(d2)
    d3 = self.reduce_matrix(d3)

    d1[np.abs(d1) < 1e-10] = 0
    d2[np.abs(d2) < 1e-10] = 0
    d3[np.abs(d3) < 1e-10] = 0

    return np.matmul(d1, xerr) + \
           0.5*np.matmul(np.matmul(d2, xerr), xerr) + \
           (1/6)*np.matmul(np.matmul(np.matmul(d3, xerr), xerr), xerr)

  def find_rho(self):
    prog = MathematicalProgram()

    # Error between state and nominal. Only consider x, y, th for the base right now, sticking to 2D
    xerr = prog.NewIndeterminates(2*(3 + self.num_rotary), 'xerr')
    xerr_dot = self.error_dynamics(-1, xerr)

    S = self.reduce_matrix(self.S)

    V = np.dot(np.dot(xerr, S), xerr)
    Vdot = np.dot(np.dot(xerr, S), xerr_dot)

    la = prog.NewSosPolynomial(Variables(xerr), 2)[0].ToExpression()

    max_improve = 10
    lower = 0
    upper = 1
    rho = upper
    i = 0
    while rho > 0:
      if i > max_improve and lower != 0:
        break

      print('Starting line search iteration %d with rho = %f' %(i, rho))

      prog_clone = prog.Clone()
      prog_clone.AddSosConstraint(-Vdot - la*(rho - V))

      result = Solve(prog_clone)

      if result.is_success():
        lower = rho
        rho = (rho + upper)/2
      else:
        upper = rho
        rho = (rho + lower)/2

      i += 1

    if lower == 0:
      print('No region of attraction')
    rho = lower
    print('Optimal rho is %f' %(rho))
