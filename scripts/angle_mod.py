import math

def fmodp(x, y):
  ret = math.fmod(x, y)
  if ret < 0:
    ret = y + ret
  return ret

def angle_mod(theta):
  return fmodp(theta, 2*math.pi)

def angdiff(th1, th2):
  th1 = angle_mod(th1)
  th2 = angle_mod(th2)

  if th2 - th1 > math.pi:
    th2 -= 2*math.pi
  elif th2 - th1 < -math.pi:
    th2 += 2*math.pi

  return th2 - th1
