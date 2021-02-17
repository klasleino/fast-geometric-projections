import numpy as np

from tensorflow.keras.utils import Progbar
from time import time

from fgp.queues import DQueue, PQueue
from fgp.utils import batch_flatten, set_hash, Timeout, TimeoutException


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

    def __repr__(self):
        return str(self)

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
    constraint. Returns the shortest distance from x to the constraint w.r.t. 
    the provided norm (implemented for l2).
    '''
    w, b = C
    return abs(np.sum(w * batch_flatten(x), axis=-1) + b) / L2(w)

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
    representation of the activation pattern when the specified neuron is 
    flipped.
    '''
    if neuron in prev_representation:
        # The neuron was already flipped, so we are unflipping it; we need to 
        # remove it from the set representation.
        return set([e for e in prev_representation if e != neuron])

    else:
        # The neuron has not been flipped so we are flipping it; we need to add
        # it to the set representation.
        return prev_representation | set([neuron])

def representation_to_activation_pattern(orig, representation):
    '''
    Given the original activation pattern of the point we are checking, returns
    the activation pattern corresponding to flipping the neurons in the given
    representation.
    '''
    return orig != np.array([n in representation for n in range(len(orig))])

def get_constraints_for_pattern(
        network, A, prev_reps, x, y, epsilon, cached=False):

    # Get the distances and constraint weights and bias.
    d, C_w, C_b = network.bprop_all(np.array(A), x, y, cached)

    # N is the number of neurons in the network.
    N = d.shape[1]

    # Collect the constraint info to be used by the algorithm. For each
    # constraint, we keep track of the distance, the neuron it is associated
    # with (or None if it is a decision boundary constraint), and the constraint
    # itself if it is a decision boundary constraint.
    constraint_info = []
    for d_i, C_wi, C_bi, A_i, prev_rep_i in zip(d, C_w, C_b, A, prev_reps):
    
        # Keep track of the neuron index each constraint is associated with. For
        # decision boundary constraints, leave this as None.
        u = [i if i < N - network.n_classes else None for i in range(N)]

        # Collect the constraints themselves. Don't keep track of constraints
        # unless they are decision boundary constraints.
        C_i = (
            [None for _ in range(N - network.n_classes)] + 
            list(zip(C_wi, C_bi)))

        # Extend `A` and `prev_reps` to the number of constraints.
        A_i = [A_i for _ in range(N)]
        prev_rep_i = [prev_rep_i for _ in range(N)]

        constraint_info_i = np.array(list(zip(d_i, u, C_i, A_i, prev_rep_i)))

        # Add the constraints that are at distance less than `epsilon` from `x`.
        constraint_info.append(constraint_info_i[d_i < epsilon])

    return np.concatenate(constraint_info)


# Because of numerical issues and arbitrary tie-breaking, if the point is 
# directly on the decision boundary, it may not classify differently (despite 
# mathematically being an adversarial example), thus we have to do a more 
# complicated check to see if the  point lies directly on the boundary.
def is_on_boundary(y_proj, pred, pred_proj, is_softmax_network=True):
    if not is_softmax_network:
        # Sigmoid networks are on (or past) the decision boundary if the probit
        # value is close to 0.5, or if the prediction on the original point is 
        # different from the prediction of the projected point.
        return np.allclose(y_proj, 0.5, atol=1e-3) or pred != pred_proj

    else:
        # Softmax networks are on (or past) the decision boundary if the
        # prediction on the original point is different from the prediction
        # of the projected point, or if the maximum predicted probit is
        # close to the second-highest predicted probit.
        return pred != pred_proj or np.allclose(
            *np.sort(y_proj)[-2:], atol=1e-3)


## Certification algorithm. ####################################################

def check(
        network, 
        x, 
        epsilon,
        timeout=120, 
        lowerbound=False, 
        keepgoing=False,
        batch_size=1,
        return_num_visited=False,
        recap=False, 
        debug_steps=False, 
        debug_print_rate=1,
        cache_first_layer=False):
    '''
    Implementation of the FGP algorithm from our paper. To use the certified 
    lower bound variant, set `lowerbound` to True. To use the heuristic 
    described in Section 2.2 for exploring the full queue to reduce the number 
    of `INCONCLUSIVE` results, set `keepgoing` to True.
    '''
    if lowerbound and batch_size > 1:
        raise ValueError(
            'Lower bound algorithm only compatible with a `batch_size` of 1.')
    if lowerbound and keepgoing:
        raise ValueError(
            'Lower bound algorithm cannot use the `keepgoing` heuristic.')
    if not network.compiled:
        raise ValueError(
            '`network` must be compiled; please call `compile_backprop`.')

    def print_recap(s):
        if debug_steps or recap:
            print(s)

    def print_debug(s):
        if debug_steps:
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
    
    # Keep track of whether the network is a sigmoid network or a softmax 
    # network.
    is_softmax_network = network.output_shape[1] > 1

    with Timeout(timeout):
        try:
            # Get the prediction for `x`.
            y_pred = (
                network.predict(np.expand_dims(x, 0)).argmax(axis=1)
                    if is_softmax_network else
                int(network.predict(np.expand_dims(x, 0))[0,0] > 0))

            # Get the activation pattern of `x`.
            x_pattern = network.get_internal_neuron_activation_pattern(x)[0]

            # We represent each activation pattern as a set of neuron indices 
            # that are different from the initial activation pattern.
            x_rep = set([])

            # Add the activation pattern for `x` (represented by the empty set)
            # to the set of visited activation patterns.
            visited.add(set_hash(x_rep))

            # Make a queue of constraints to check and add the nearby 
            # constraints to `x`.
            queue = (
                PQueue(visited) if lowerbound else 
                DQueue(visited, batch_size=batch_size))

            # Initialize the queue.
            for d, u, C, A, rep in get_constraints_for_pattern(
                    network, 
                    [x_pattern], # List of 1 activation pattern
                    [x_rep], # List of 1 set
                    x,
                    y_pred,
                    epsilon):
                
                if u is None:
                    # This is a decision boundary constraint.
                    queue.put((d, C, A, None))

                else:
                    queue.put((d, C, A, flip_neuron(x_rep, u)))

            print_recap('Started with {} constraints to check.'.format(
                queue.qsize()))

            while not queue.empty():
                # Dequeue a batch of constraints.
                constrs = queue.get()
                
                if not constrs:
                    # This means we did not dequeue any constraints leading to
                    # regions we have not yet visited.
                    continue
                
                if lowerbound:
                    # Note that in this case `len(constrs) == 1`.
                    d, _, _, _ = constrs[0]
                    if d > eps_lowerbound:
                        print_recap(d)
                    eps_lowerbound = max(eps_lowerbound, d)

                if constrs[0][-1] is None:
                    # All the constraints are decision boundaries.
                    for d, C, A, rep in constrs:

                        print_debug(
                            'Found decision boundary within epsilon radius.')
                        print_debug('Checked {} regions.'.format(len(visited)))

                        # Check if the projected point is in fact an adversarial
                        # example.
                        x_proj = projection(x.flatten(), C)

                        if not is_softmax_network:
                            y = network.predict(np.expand_dims(x, 0))[0,0]
                            y_proj = network.predict(
                                np.expand_dims(x_proj.reshape(x.shape), 0))[0,0]
                            pred = int(y > 0.5)
                            pred_proj = int(y_proj > 0.5)
                        else:
                            y = network.predict(np.expand_dims(x, 0))[0]
                            y_proj = network.predict(
                                np.expand_dims(x_proj.reshape(x.shape), 0))[0]
                            pred = y.argmax()
                            pred_proj = y_proj.argmax()

                        # Because of numerical issues and arbitrary 
                        # tie-breaking, if the point is directly on the decision
                        # boundary, it may not classify differently (despite 
                        # mathematically being an adversarial example), thus we
                        # have to do a more complicated check to see if the 
                        # point lies directly on the boundary.
                        if is_on_boundary(
                                y_proj, pred, pred_proj, is_softmax_network):

                            print_recap(
                                'Found a true adversarial example at distance '
                                '{:.4f}.'
                                .format(d))

                            return result(NOT_ROBUST)

                        else:
                            # NOTE: here we could fall back on constraint 
                            #   solving to determine conclusively if this is a
                            #   true or false positive.

                            if keepgoing and not lowerbound:
                                # We cannot conclude anything about this 
                                # decision boundary, but we keep the search in 
                                # case we find a true adversarial example. If we
                                # don't, we will return inconclusive.
                                has_unknown = True
                                continue

                            else:
                                print_recap(
                                    'Inconclusive analysis. Robustness '
                                    'unknown.')
                                if lowerbound:
                                    print_recap(
                                        'Proven roubst up to epsilon = {:.4f}'
                                        .format(eps_lowerbound))

                                return result(INCONCLUSIVE)

                else:
                    # Visit the activation pattern and add its constraints to 
                    # the queue.

                    if debug_steps and len(visited) % debug_print_rate == 0:
                        print(
                            'Visiting activation pattern {}. Visted {} regions '
                            'so far.'
                            .format(rep, len(visited)))

                    representations = [rep for d, C, A, rep in constrs]

                    patterns = [
                        representation_to_activation_pattern(x_pattern, rep)
                        for rep in representations]

                    for d, u, C, A, rep in get_constraints_for_pattern(
                            network, 
                            patterns,
                            representations, 
                            x, 
                            y_pred,
                            epsilon,
                            cached=cache_first_layer):

                        if u is None:
                            # This is a decision boundary constraint.
                            queue.put((d, C, A, None))

                        else:
                            new_rep = flip_neuron(rep, u)
                            queue.put((d, C, A, new_rep))


                    if debug_steps and len(visited) % debug_print_rate == 0:
                        print('Now have {} constraints to check.'.format(
                            queue.qsize()))

            # If we finish searching all nearby constraints and haven't found a
            # decision boundary, we are done and the network is robust at this 
            # point.
            print_recap('No more constraints to check.')
            print_recap('Checked {} regions.'.format(len(visited)))

            if has_unknown:
                # We finished the queue, but had previously an inconclusive 
                # decision boundary.
                print_recap(
                    'Explored all constraints in queue, but previously had an '
                    'inconclusive decision boundary.')

                return result(INCONCLUSIVE)

            if lowerbound:
                eps_lowerbound = epsilon

            return result(ROBUST)

        except TimeoutException:
            print_recap('Timed out.')
            if lowerbound:
                print_recap(
                    'Proven roubst up to epsilon = {:.4f}'.format(
                        eps_lowerbound))

            return result(TIMED_OUT)


def eval_verification(
        m, 
        N, 
        x_test, 
        y_test, 
        epsilon=0.25, 
        batch_size=100, 
        timeout=60, 
        seed=0):
    
    np.random.seed(0)

    n_robust = 0
    n_nonrobust = 0
    n_timeout = 0
    n_unknown = 0
    n_robust_correct = 0
    times = []
    n_visited = []
    pb = Progbar(N, stateful_metrics=['vra'])

    for j in range(N):
        start_time = time()

        r = check(
            m, 
            x_test[j], 
            epsilon, 
            recap=False, 
            timeout=timeout, 
            keepgoing=True,
            return_num_visited=True,
            batch_size=batch_size,
            lowerbound=False)

        end_time = time()

        n_visited.append(r[1])

        if r[0] is ROBUST:
            n_robust += 1
            if m.predict(x_test[j:j+1]).argmax(axis=1)[0] == y_test[j].argmax():
                n_robust_correct += 1

        elif r[0] is INCONCLUSIVE:
            n_unknown += 1

        elif r[0] is NOT_ROBUST:
            n_nonrobust += 1

        if r[0] is TIMED_OUT:
            n_timeout += 1

        else:
            times.append(end_time - start_time)

        pb.add(1, [
            ('vra', float(n_robust_correct)/float(j+1)),
            ('ro', 1 if r[0] is ROBUST else 0), 
            ('adv', 1 if r[0] is NOT_ROBUST else 0), 
            ('unk', 1 if r[0] is INCONCLUSIVE else 0), 
            ('to', 1 if r[0] is TIMED_OUT else 0),
            ('rt', end_time-start_time),
            ('vis', r[1])])

    print('# robust: {}'.format(n_robust))
    print('# non-robust: {}'.format(n_nonrobust))
    print('# timeout: {}'.format(n_timeout))
    print('# unknown: {}'.format(n_unknown))
    print(
        'med time: {:.2f}'
        .format(np.sort(np.array(times))[int(len(times) / 2)]))
    print('mean time: {:.2f}'.format(np.array(times).mean()))
    print('mean # visited: {:.1f}'.format(np.array(n_visited).mean()))
    print(
        'median # regions visited: {:.1f}'
        .format(np.sort(np.array(n_visited))[int(len(n_visited) / 2)]))
    print(
        'verified robust accuracy: {:.2}'
        .format(float(n_robust_correct)/float(N)))
