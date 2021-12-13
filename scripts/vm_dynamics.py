import sympy
from sympy.physics.vector import dynamicsymbols
import pickle
import numpy as np

class VMDynamics(object):
  def __init__(self):
    self.num_rotary = 3
    self.num_links = self.num_rotary + 1

    self.masses = [10 for l in range(self.num_links)] # All links have center of mass at geometric center
    self.Mtot = sum(self.masses)
    self.inertias = [1.67] + [0.0125 for l in range(1, self.num_links)]
    self.lengths = [1] + [1 for l in range(1, self.num_links)]
    self.Rs = [sympy.ImmutableMatrix([[self.lengths[l]/2], [0]]) for l in range(self.num_links)] # Starting at base link (link 0)
    self.Ls = [sympy.ImmutableMatrix([[self.lengths[l]/2], [0]]) for l in range(1, self.num_links)] # Starting at link 1
    self.rs = [self.Rs[l]*sum(self.masses[:l + 1])/self.Mtot for l in range(self.num_links)] # Starting at link 0
    self.ls = [self.Ls[l - 1]*sum(self.masses[:l])/self.Mtot for l in range(1, self.num_links)] # Starting at link 1
    self.Ws = [] # Define VM going to the CoM of each link
    for l in range(self.num_links):
      self.Ws.append([])
      for lp in range(self.num_links):
        if lp < l:
          if lp == 0:
            self.Ws[-1].append(self.rs[0])
          else:
            self.Ws[-1].append(self.rs[lp] + self.ls[lp - 1])
        elif lp == l:
          if lp == 0:
            self.Ws[-1].append(self.rs[0] - self.Rs[0])
          else:
            self.Ws[-1].append(self.rs[lp] + self.ls[lp - 1] - self.Rs[lp])
        else:
          self.Ws[-1].append(self.rs[lp] + self.ls[lp - 1] - self.Rs[lp] - self.Ls[lp - 1])

    self.ee_Ws = [W for W in self.Ws[-1]]
    self.ee_Ws[-1] += self.Rs[-1]

    self.t = None
    self.q = None
    self.qdot = None
    self.qddot = None
    self.tau = None
    self.eqns = None
    self.trig_eqns = None
    self.c = None
    self.s = None
    
  def euler_lagrange(self):
    q_vars = ['phi'] + ['th' + str(i + 1) for i in range(self.num_rotary)]
    qdot_vars = ' '.join([var + 'dot' for var in q_vars])
    qddot_vars = ' '.join([var + 'ddot' for var in q_vars])
    q_vars = ' '.join(q_vars)
    self.q = list(dynamicsymbols(q_vars))

    self.t = sympy.Symbol('t')
    qdot = [sympy.diff(var, self.t) for var in self.q]
    qddot = [sympy.diff(var, self.t) for var in qdot]

    L = self.KE(self.q, qdot)

    dLdq = [sympy.diff(L, var) for var in self.q]
    dLdqdot = [sympy.diff(L, var) for var in qdot]
    d2Ldqdotdt = [sympy.diff(d, self.t) for d in dLdqdot]
    print('Got derivatives')

    self.tau = [0] + sympy.symbols(['tau' + str(i + 1) for i in range(self.num_rotary)])
    self.eqns = [term1 - term2 - term3 for term1, term2, term3 in zip(d2Ldqdotdt, dLdq, self.tau)]
    print('Got equations')

    self.qdot = list(dynamicsymbols(qdot_vars))
    qddot_tmp = [sympy.diff(var, self.t) for var in self.qdot]
    self.qddot = list(sympy.symbols(qddot_vars))
    for i in range(len(self.eqns)):
      for old, new in zip(qdot, self.qdot): 
        self.eqns[i] = self.eqns[i].subs(old, new)

      for old, new in zip(qddot_tmp, self.qddot): 
        self.eqns[i] = self.eqns[i].subs(old, new)

      for old, new in zip(qddot, self.qddot):
        self.eqns[i] = self.eqns[i].subs(old, new)

    print('Substituted new variables')

  def KE(self, q, qdot):
    T = 0
    # Iterate through the VMs. EE velocity of the VM
    # equals the COM velocity of the link it terminates at
    for l in range(self.num_links):
      # Iterate through the VM links
      angle_sum = 0
      com_pos = sympy.zeros(2, 1)
      for lp in range(self.num_links):
        angle_sum += q[lp]
        c = sympy.cos(angle_sum)
        s = sympy.sin(angle_sum)
        R = sympy.ImmutableMatrix([[c, -s], [s, c]])
        com_pos += R*self.Ws[l][lp]

      com_vel = sympy.diff(com_pos, self.t)
      ang_vel = sum(qdot[:l + 1]) 
      T += 0.5*(self.masses[l]*(com_vel.transpose()*com_vel)[0, 0] + self.inertias[l]*ang_vel**2)

    return T

  def get_qddot(self, q, qdot, u):
    new_eqns = [eqn for eqn in self.eqns]
    for i in range(len(q)):
      for j in range(len(self.eqns)):
        new_eqns[j] = new_eqns[j].subs(self.q[i], q[i])
        new_eqns[j] = new_eqns[j].subs(self.qdot[i], qdot[i])

    for i in range(len(u)):
      for j in range(len(self.eqns)):
        new_eqns[j] = new_eqns[j].subs(self.tau[i + 1], u[i])

    qddot = sympy.solve(new_eqns, self.qddot)
    ret = np.zeros(len(q))
    for i, var in enumerate(self.qddot):
      ret[i] = qddot[var]

    return ret

  def linearize(self, q, qdot, u):
    eps = 0.001
    nq = len(q)
    nu = len(u)

    qddot = self.get_qddot(q, qdot, u)

    A = np.zeros((2*nq, 2*nq))
    A[:nq, nq:] = np.eye(nq)
    for i in range(2*nq):
      if i < nq:
        q1 = np.copy(q)
        q1[i] += eps
        qdot1 = qdot
      else:
        qdot1 = np.copy(qdot)
        qdot1[i - nq] += eps
        q1 = qdot

      qddot1 = self.get_qddot(q1, qdot1, u)
      A[nq:, i] = (qddot1 - qddot)/eps

    B = np.zeros((2*nq, nu))
    for i in range(nu):
      u1 = np.copy(u)
      u1[i] += eps

      qddot1 = self.get_qddot(q, qdot, u1)
      B[nq:, i] = (qddot1 - qddot)/eps

    return A, B

  def ee_Jacobian(self, q):
    ee_pos = sympy.zeros(2, 1)
    angle_sum = 0
    for l in range(self.num_links):
      angle_sum += self.q[l]
      c = sympy.cos(angle_sum)
      s = sympy.sin(angle_sum)
      R = sympy.ImmutableMatrix([[c, -s], [s, c]])
      ee_pos += R*self.ee_Ws[l]

    J = sympy.zeros(len(ee_pos), len(q))
    for i in range(len(ee_pos)):
      for j in range(len(q)):
        J[i, j] = sympy.diff(ee_pos[i], self.q[j])

    for j in range(len(q)):
      J = J.subs(self.q[j], q[j])
    
    return np.array(J).astype(np.float64)

  def linearize_ee(self, q, qdot, u):
    A, B = self.linearize(q, qdot, u)

    ee_pos = sympy.zeros(2, 1)
    angle_sum = 0
    for l in range(self.num_links):
      angle_sum += self.q[l]
      c = sympy.cos(angle_sum)
      s = sympy.sin(angle_sum)
      R = sympy.ImmutableMatrix([[c, -s], [s, c]])
      ee_pos += R*self.ee_Ws[l]

    J = sympy.zeros(len(ee_pos), len(q))
    for i in range(len(ee_pos)):
      for j in range(len(q)):
        J[i, j] = sympy.diff(ee_pos[i], self.q[j])

    Jdot = sympy.diff(J, self.t)
    qdot_tmp = [sympy.diff(qi, self.t) for qi in self.q]
    for j in range(len(q)):
      J = J.subs(self.q[j], q[j])
      Jdot = Jdot.subs(qdot_tmp[j], qdot[j])
      Jdot = Jdot.subs(self.q[j], q[j])

    J = np.array(J).astype(np.float64)
    Jdot = np.array(Jdot).astype(np.float64)

    A = np.concatenate((A, np.zeros((4, A.shape[1]))))
    A = np.concatenate((A, np.zeros((A.shape[0], 4))), 1)
    B = np.concatenate((B, np.zeros((4, B.shape[1]))))

    A[2*len(q):2*len(q) + 2, len(q):2*len(q)] = J
    A[2*len(q) + 2:2*len(q) + 4, len(q):2*len(q)] = Jdot
    A[2*len(q) + 2:2*len(q) + 4] += np.matmul(J, A[len(q):2*len(q)])
    B[2*len(q) + 2:2*len(q) + 4] = np.matmul(J, B[len(q):2*len(q)])

    return A, B

  def gen_trig_dynamics(self):
    cs = [sympy.cos(qi) for qi in self.q]
    ss = [sympy.sin(qi) for qi in self.q]
    c_vars = ['c0'] + ['c' + str(i + 1) for i in range(self.num_rotary)]
    c_vars = ' '.join(c_vars)
    s_vars = ['s0'] + ['s' + str(i + 1) for i in range(self.num_rotary)]
    s_vars = ' '.join(s_vars)
    self.c = sympy.sympy.symbols(c_vars)
    self.s = sympy.sympy.symbols(s_vars)
    self.trig_eqns = [eqn for eqn in self.eqns]
    for i in range(len(self.eqns)):
      for old, new in zip(cs, self.c): 
        self.trig_eqns[i] = self.trig_eqns[i].subs(old, new)
      for old, new in zip(ss, self.s): 
        self.trig_eqns[i] = self.trig_eqns[i].subs(old, new)

    print('Generated trig equations')
