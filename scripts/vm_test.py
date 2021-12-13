from vm_dynamics import VMDynamics
import numpy as np
from scipy.linalg import solve_continuous_are

vmd = VMDynamics()
vmd.euler_lagrange()

nq = 4
q = np.zeros(nq)
qdot = np.zeros(nq)
u = np.zeros(nq - 1)
nu = len(u)
qddot = vmd.get_qddot(q, qdot, u)

eps = 0.001
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

  qddot1 = vmd.get_qddot(q1, qdot1, u)
  A[nq:, i] = (qddot1 - qddot)/eps

B = np.zeros((2*nq, nu))
for i in range(nu):
  u1 = np.copy(u)
  u1[i] += eps

  qddot1 = vmd.get_qddot(q, qdot, u1)
  B[nq:, i] = (qddot1 - qddot)/eps

Q = np.eye(2*nq)
R = np.eye(nu)
S = solve_continuous_are(A, B, Q, R)
print(S)
