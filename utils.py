import signal


def set_hash(s):
  '''
  Hashes a set in a way that is consistent for sets containing the same 
  elements.
  '''
  return str(sorted(list(s)))


class TimeoutException(Exception):
  pass
    
class Timeout(object):
  '''
  Allows us to let the algorithm time out rather than run indefinitely on inputs
  that take a long time to certify.
  '''
  def __init__(self, seconds):
    self.seconds = seconds

  def handle_timeout(self, signum, frame):
    raise TimeoutException()

  def __enter__(self):
    if self.seconds is not None:
      signal.signal(signal.SIGALRM, self.handle_timeout)
      signal.alarm(self.seconds)

  def __exit__(self, type, value, traceback):
    if self.seconds is not None:
      signal.alarm(0)


def sat(x, C):
  ''' Checks if a given point satisfies the given constraint.'''
  w, b = C
  return w.dot(x) + b < 0
