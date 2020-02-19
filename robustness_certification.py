import numpy as np

from itertools import count

from queues import DQueue, PQueue
from utils import set_hash, Timeout, TimeoutException


# Define result type.

class _RobustnessResult(object):
  def __init__(self, enum_id):
    self.enum_id = enum_id

  def __eq__(self, other):
    if isinstance(other, _RobustnessResult):
      return other.enum_id == self.enum_id
    else:
      return False

  def __hash__(self):
    return hash(self.enum_id)

  def __bool__(self):
    return self.enum_id == 'ROBUST'

  def __str__(self):
    return self.enum_id

# Define robustness results.
NOT_ROBUST = _RobustnessResult('NOT_ROBUST')
ROBUST = _RobustnessResult('ROBUST')
INCONCLUSIVE = _RobustnessResult('INCONCLUSIVE')
TIMED_OUT = _RobustnessResult('TIMED_OUT')


# Here we implement projections and the projection-based distance metric for
# taking the distrance from a point to a constraint. The implementations here
# are specifically for the l2 norm.

# Define the l2 norm.
L2 = lambda x: np.sqrt(np.sum(x * x, axis=-1))

def constraint_distance(x, C):
  '''
  `x` is a point. `C`, given as w'X + b is a hyperplane representing a 
  constraint. Returns the shortest distance from x to the constraint w.r.t. the 
  provided norm (implemented for l2).
  '''
  w, b = C
  return abs(np.sum(w * x, axis=-1) + b) / L2(w)

def projection(x, C, delta=0.):
  '''
  Returns a point in `C` with minimal distance to `x`, where `C` is the half-
  space, `w'X + b` given as the tuple `(w, b)` (implemented for the l2 norm). 
  '''
  w, b = C
  w_dot_x_plus_b = w.dot(x) + b
  return x - (w_dot_x_plus_b) * w / L2(w)**2


# We represent an activation pattern in the visited set as a set of integer
# indices keeping track of which neurons have been flipped from the algorithms's
# initial activation pattern. Here we define some utility functions for
# converting between this sparse representation and the representation used on
# the GPU, which is simply a float32 array of ones and zeros.

def flip_neuron(prev_representation, neuron):
  '''
  Given the representation of the previous activation pattern, returns the
  representation of the activation pattern when the specified neuron is flipped.
  '''
  if neuron in prev_representation:
    # The neuron was already flipped, so we are unflipping it; we need to remove 
    # it from the set representation.
    return set([e for e in prev_representation if e != neuron])

  else:
    # The neuron has not been flipped so we are flipping it; we need to add it
    # to the set representation.
    return prev_representation | set([neuron])

def representation_to_activation_pattern(orig, representation):
  '''
  Given the original activation pattern of the point we are checking, returns
  the activation pattern corresponding to flipping the neurons in the given
  representation.
  '''
  return orig != np.array([n in representation for n in range(len(orig))])

def get_constraints_for_pattern(
    network, 
    patterns, 
    pattern_representations,
    x,
    y, 
    epsilon, 
    num_boundary_constraints=1):

  # Computing distances on the GPU was only implemented for softmax models
  # (though it will not be difficult to implement this for sigmoid models as
  # well).
  distances_on_gpu = num_boundary_constraints > 1

  if distances_on_gpu:
    batched_constraints = network.bprop_distances(np.array(patterns), x, c=y)

    neuron_constraints = batched_constraints[:-num_boundary_constraints:]
    output_constraints = batched_constraints[-num_boundary_constraints:]

    # Keep track of the neuron corresponding to each constraint. The constraints
    # are listed in order of their internal neuron index. The last 
    # `num_boundary_constraints` constraints returned by `bprop_cosntraints` 
    # are always the decision boundary constraints; this is indicated by having 
    # a neuron index of `None`.
    output_constraints = [
      (constraint_distance(x, c), (c, None)) for c in output_constraints]
    neuron_constraints = [
      (d, (None, n)) for n, d in enumerate(neuron_constraints)]

  else:
    batched_constraints = network.bprop_cosntraints(np.array(patterns), c=y)

    # Keep track of the neuron corresponding to each constraint. The constraints
    # are listed in order of their internal neuron index. The last 
    # `num_boundary_constraints` constraints returned by `bprop_cosntraints` 
    # are always the decision boundary constraints; this is indicated by having 
    # a neuron index of `None`.
    batched_constraints = (
      [(c, None) for c in batched_constraints[-num_boundary_constraints:]] +
      [(c, n) for n, c in enumerate(
        batched_constraints[:-num_boundary_constraints])])

    # Calculate the distance to the constraints since it was not done on the 
    # GPU.
    batched_constraints = [
      (constraint_distance(x, c), (c, n))
      for c, n in batched_constraints]

  # Gather the constraints, and their associated representations and distances,
  # from the batched form returned by the GPU.
  constraints = []
  for i, (pattern, rep) in enumerate(zip(patterns, pattern_representations)):
    constraints += [
      (d[i], rep, ((c[0][i], c[1][i]), n, pattern))
      for (d, (c, n)) in output_constraints
      if d[i] < epsilon]
    constraints += [
      (d[i], rep, (None, n, pattern))
      for (d, (c, n)) in neuron_constraints
      if d[i] < epsilon]

  return constraints


## Certification algorithm. ####################################################

def check(
    network, 
    x, 
    epsilon,
    timeout=None, 
    lowerbound=False, 
    keepgoing=False,
    batch_size=1,
    return_num_visited=False,
    recap=False, 
    debug_steps=False, 
    debug_print_rate=1):
  '''
  Implementation of Algorithm 1 and Algorithm 2 from our paper. To use Algorithm
  2 (the certified lower bound variant), set `lowerbound` to True. To use the
  heuristic described in Section 3.3 for exploring the full queue to reduce the
  number of `INCONCLUSIVE` results, set `keepgoing` to True.
  '''
  if lowerbound and batch_size > 1:
    raise ValueError(
      'Lower bound algorithm only compatible with a `batch_size` of 1.')
  if lowerbound and keepgoing:
    raise ValueError(
      'Lower bound algorithm cannot use the `keepgoing` heuristic.')

  def print_recap(s):
    if debug_steps or recap:
      print(s)

  # Used when calculating the certified lower bound.
  eps_lowerbound = 0.

  # Used when `keepgoing` is set to True.
  has_unknown = False

  # Keeps track of the regions we've visited in our search.
  visited = set()

  def result(r):
    if return_num_visited:
      return (
        (r, eps_lowerbound, len(visited)) if lowerbound else 
        (r, len(visited)))
    else:
      return (r, eps_lowerbound) if lowerbound else r

    return (r, eps) if lowerbound else (r, nv) if return_num_visited else r

  # Get the number of boundary constraints for the network. If the network is a
  # softmax model, it will have a `n_classes` field; otherwise it is a sigmoid
  # network, which has only one boundary.
  try:
    num_boundary_constraints = network.n_classes - 1
  except:
    num_boundary_constraints = 1

  with Timeout(timeout):
    try:
      # Get the prediction for `x`.
      y_pred = (
        int(network.predict(np.expand_dims(x, 0)).argmax())
          if num_boundary_constraints > 1 else
        int(network.predict(np.expand_dims(x, 0))[0,0] > 0))

      # Get the activation pattern of `x`.
      x_pattern = network.get_internal_neuron_activation_pattern(x)[0]

      # We represent each activation pattern as a set of neuron indices that are
      # different from the initial activation pattern.
      x_rep = set([])

      # Add the activation pattern for `x` (represented by the empty set) to the 
      # set of visited activation patterns.
      visited.add(set_hash(x_rep))

      # Make a queue of constraints to check and add the nearby constraints to 
      # `x`.
      queue = (
        PQueue(visited) if lowerbound else 
        DQueue(visited, batch_size=batch_size))

      # Initialize the queue.
      unique = count()
      for dist, rep, (c, n, act) in get_constraints_for_pattern(
          network, 
          [x_pattern],
          [x_rep], 
          x,
          y_pred, 
          epsilon,
          num_boundary_constraints):
        
        if n is None:
          # This is a decision boundary constraint.
          queue.put((dist, next(unique), (c, None, act)), True)

        else:
          queue.put(
            (dist, next(unique), (c, flip_neuron(x_rep, n), act)), 
            False)

      print_recap('Started with {} constraints to check.'.format(queue.qsize()))

      while not queue.empty():
        # Dequeue a batch of constraints.
        constrs = queue.get()

        if not constrs:
          # This means we did not dequeue any constraints leading to regions we
          # have not yet visited.
          continue
        
        if lowerbound:
          if constrs[0][0] > eps_lowerbound:
            print_recap(constrs[0][0])
          eps_lowerbound = max(eps_lowerbound, constrs[0][0])

        if constrs[0][2][1] is None:
          # All the constraints are decision boundaries.
          for dist, u, (c, rep, act) in constrs:

            print_recap('Found decision boundary within epsilon radius.')
            print_recap('Checked {} regions.'.format(len(visited)))

            # Check if the projected point is in fact an adversarial example.
            x_proj = projection(x, c)

            if num_boundary_constraints == 1:
              y = network.predict(np.expand_dims(x, 0))[0,0]
              y_proj = network.predict(np.expand_dims(x_proj, 0))[0,0]
            else:
              y = network.predict(np.expand_dims(x, 0))[0]
              y_proj = network.predict(np.expand_dims(x_proj, 0))[0]
              pred = network.predict(np.expand_dims(x, 0)).argmax()
              pred_proj = network.predict(np.expand_dims(x_proj, 0)).argmax()

            # Because of numerical issues and arbitrary tie-breaking, if the 
            # point is directly on the decision boundary, it may not classify
            # differently (despite mathematically being an adversarial example),
            # thus we have to do a more complicated check to see if the point
            # lies directly on the boundary.
            if (
                num_boundary_constraints == 1 and (
                  (np.allclose(y_proj, 0.5, atol=1e-3) or 
                  (y > 0.5 and y_proj < 0.5) or 
                  (y < 0.5 and y_proj > 0.5)))) or (
                num_boundary_constraints > 1 and pred != pred_proj):

              print_recap(
                'Found a true adversarial example at distance {:.4f}.'
                .format(dist))

              return result(NOT_ROBUST)

            else:
              # NOTE: here we could fall back on constraint solving as decribed
              #   in Section 3.3 to determine conclusively if this is a true or
              #   false positive.

              if keepgoing and not lowerbound:
                # We cannot conclude anything about this decision boundary, but 
                # we keep the search in case we find a true adversarial example.
                # If we don't, we will return inconclusive.
                has_unknown = True
                continue

              else:
                print_recap('Inconclusive analysis. Robustness unknown.')
                if lowerbound:
                  print_recap(
                    'Proven roubst up to epsilon = {:.4f}'
                    .format(eps_lowerbound))

                return result(INCONCLUSIVE)

        else:
          # Visit the activation pattern and add its constraints to the queue.

          if debug_steps and len(visited) % debug_print_rate == 0:
            print(
              'Visiting activation pattern {}. Visted {} regions so far.'
              .format(rep, len(visited)))

          patterns = [
            representation_to_activation_pattern(x_pattern, rep)
            for dist, u, (c, rep, act) in constrs]

          representations = [rep for dist, u, (c, rep, act) in constrs]

          for dist, rep, (c, n, act) in get_constraints_for_pattern(
              network, 
              patterns,
              representations, 
              x, 
              y_pred,
              epsilon,
              num_boundary_constraints):

            if n is None:
              # This is a decision boundary constraint.
              queue.put((dist, next(unique), (c, None, act)), True)

            else:
              queue.put(
                (dist, next(unique), (c, flip_neuron(rep, n), act)), 
                False)

          if debug_steps and len(visited) % debug_print_rate == 0:
            print('Now have {} constraints to check.'.format(queue.qsize()))

      # If we finish searching all nearby constraints and haven't found a
      # decision boundary, we are done and the network is robust at this point.
      print_recap('No more constraints to check.')
      print_recap('Checked {} regions.'.format(len(visited)))

      if has_unknown:
        # We finished the queue, but had previously an inconclusive decision 
        # boundary.
        print_recap(
          'Explored all constraints in queue, but previously had an '
          'inconclusive decision boundary')

        return result(INCONCLUSIVE)

      if lowerbound:
        eps_lowerbound = epsilon

      return result(ROBUST)

    except TimeoutException:
      print_recap('Timed out.')
      if lowerbound:
        print_recap(
          'Proven roubst up to epsilon = {:.4f}'.format(eps_lowerbound))

      return result(TIMED_OUT)
