import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf


def proj_dist(proj_type, x, grads, biases):
    if proj_type == 'l2':
        return l2_proj_distance(x, grads, biases)

    elif proj_type == 'linf':
        return linf_proj_distance(x, grads, biases)

    else:
        raise ValueError('projection type "{}" not supported'.format(proj_type))

def l2_proj_distance(x, grads, biases):
    return (K.abs(K.sum(grads * x[:,None], axis=-1) + biases) / 
        K.sqrt(K.sum(grads * grads, axis=-1)))

def linf_proj_distance(x, grads, biases):
    return (K.abs(K.sum(grads * x[:,None], axis=-1) + biases) /
        K.sum(K.abs(grads), axis=-1))



# NOT ACTUALLY USED IN PAPER ###################################################

# Projections constrained to the [0, 1] cube can be computed reasonably 
# efficiently using Lagrange multipliers. While this is cheaper than solving
# LPs/QPs, it is still expensive, and we find that it does not reduce the number
# of regions searched sufficiently to make up for the extra time; that is, the
# FGP algorithm is faster without this addition.

def constrained_l2_proj_distance_save_memory(x, grads, biases):
    u, w, b = x, grads, -biases[:,:,None]

    ndim = K.int_shape(u)[1]
    neurons = K.int_shape(w)[1]
    w = K.reshape(w, (-1, ndim))
    b = K.reshape(b, (-1, 1))

    # Handy naming to make code a bit more readable.
    def is_(x):
        return K.cast(x, 'float32')
    def where(x):
        return K.cast(x, 'float32')

    # Check feasibility.
    # The problem is infeasible if `q` (defined below) can be brought to 
    # infinity as `lam` -> +/-infinity. We see that as `lam` -> +infinity, 
    # `x_star[i]` becomes 0 if `w[i]` is positive and if `w[i]` is negative. 
    # Thus, dq/dlam = -b + sum_{i : w_i < 0}{w_i}. If dq/dlam is positive, then
    # `q` will go to infinity. The other case comes from the symmetric case for
    # when `lam` -> -infinity.
    infeasible = (
        is_(K.sum(w * where(w > 0), axis=1) < b[:,0]) + 
        is_(K.sum(w * where(w < 0), axis=1) > b[:,0]))

    feasible = 1. - infeasible

    # Get the order (as lambda goes from -infinity to 0) in which each dimention 
    # transitions to the I stage.
    I_in_order = tf.argsort(
        u / w * where(w < 0) + 
        (u - 1) / w * where(w > 0))

    # Get the order (as lambda goes from 0 to +infinity) in which each dimention 
    # transitions out of the I stage.
    I_out_order = tf.argsort(
        u / w * where(w > 0) + 
        (u - 1) / w * where(w < 0))

    w_I_in = tf.gather(w, I_in_order, batch_dims=1)
    u_I_in = tf.gather(u, I_in_order, batch_dims=1)
    w_1_in = tf.gather(w * where(w > 0), I_in_order, batch_dims=1)

    in_nums = w_I_in * u_I_in - w_1_in
    in_denoms = w_I_in**2

    w_I_out = tf.gather(w, I_out_order, batch_dims=1)
    u_I_out = tf.gather(u, I_out_order, batch_dims=1)
    w_1_out = tf.gather(w * where(w < 0), I_out_order, batch_dims=1)
    
    out_nums = -w_I_out * u_I_out + w_1_in
    out_denoms = -w_I_out**2

    nums = (
        K.sum(w * where(w > 0), axis=1)[:,None] + 
            K.cumsum(
                K.concatenate((in_nums, out_nums), axis=1),
                axis=1)[:,:-1] -
            b)

    denoms = K.cumsum(
        K.concatenate((in_denoms, out_denoms), axis=1),
        axis=1)[:,:-1]
    
    argmaxes = nums / denoms

    # Find the inflection points in `q`.
    inflections = K.concatenate((
        (u - 1) / w, 
        u / w))

    lam = K.concatenate((argmaxes, inflections))

    i0 = tf.constant(0)
    m0 = K.zeros((0,ndim))

    loop_condition = lambda i, m: i < K.shape(lam)[0]

    def loop_body(i, m):
        x_star_candidate_i = K.clip(
            u[0,None] - lam[i,:,None] * w[i,None], 0., 1.)

        max_q_candidate_i = (K.sum(
            .5 * (x_star_candidate_i - u[0,None])**2 + 
                lam[i,:,None] * w[i,None] * x_star_candidate_i,
            axis=-1) - lam[i] * b[i])

        opt_i = K.cast(K.argmax(max_q_candidate_i, axis=0), 'int32')

        return [
            i+1,
            tf.concat((m, x_star_candidate_i[opt_i,None]), axis=0)]
    
    x_star = tf.while_loop(
        loop_condition, 
        loop_body, 
        loop_vars=[i0, m0],
        shape_invariants=[i0.get_shape(), tf.TensorShape([None,ndim])])[1]

    d = tf.norm(x_star - u, axis=1) / feasible
    
    return K.reshape(d, (-1, neurons))

def constrained_l2_proj_distance(x, grads, biases):
    u, w, b = x, grads, -biases[:,:,None]

    ndim = K.int_shape(u)[1]
    neurons = K.int_shape(w)[1]
    w = K.reshape(w, (-1, ndim))
    b = K.reshape(b, (-1, 1))

    # Handy naming to make code a bit more readable.
    def is_(x):
        return K.cast(x, 'float32')
    def where(x):
        return K.cast(x, 'float32')

    # Check feasibility.
    # The problem is infeasible if `q` (defined below) can be brought to 
    # infinity as `lam` -> +/-infinity. We see that as `lam` -> +infinity, 
    # `x_star[i]` becomes 0 if `w[i]` is positive and if `w[i]` is negative. 
    # Thus, dq/dlam = -b + sum_{i : w_i < 0}{w_i}. If dq/dlam is positive, then
    # `q` will go to infinity. The other case comes from the symmetric case for
    # when `lam` -> -infinity.
    infeasible = (
        is_(K.sum(w * where(w > 0), axis=1) < b[:,0]) + 
        is_(K.sum(w * where(w < 0), axis=1) > b[:,0]))

    feasible = 1. - infeasible

    # Get the order (as lambda goes from -infinity to 0) in which each dimention 
    # transitions to the I stage.
    I_in_order = tf.argsort(
        u / w * where(w < 0) + 
        (u - 1) / w * where(w > 0))

    # Get the order (as lambda goes from 0 to +infinity) in which each dimention 
    # transitions out of the I stage.
    I_out_order = tf.argsort(
        u / w * where(w > 0) + 
        (u - 1) / w * where(w < 0))

    eye = K.eye(ndim)

    mask_I = K.cumsum(
        K.concatenate(
            (
                K.gather(eye, I_in_order),
                -K.gather(eye, I_out_order)),
            axis=1),
        axis=1)

    mask_I = tf.gather(mask_I, np.arange(2*ndim - 1), axis=1)

    mask_1 = is_(w > 0)[:,None] + K.cumsum(
        K.concatenate(
            (
                -tf.gather(
                    eye[None] * where(w > 0)[:,None], I_in_order, batch_dims=1),
                tf.gather(
                    eye[None] * where(w < 0)[:,None], 
                    I_out_order, 
                    batch_dims=1)),
            axis=1),
        axis=1)

    mask_1 = tf.gather(mask_1, np.arange(2*ndim - 1), axis=1)

    # Here we collect the dimensions in `w` and `u` that are in the I and 1
    # phase in state `i`. An analysis of the cases shows that given these
    # dimension, the argmax of `q` can be found by the equation below.
    w_I = w[:,None] * mask_I
    u_I = u[:,None] * mask_I
    w_1 = w[:,None] * mask_1

    argmaxes = (
        (K.sum(w_I * u_I, axis=2) + K.sum(w_1, axis=2) - b) /
        K.sum(w_I * w_I, axis=2))

    # Find the inflection points in `q`.
    inflections = K.concatenate((
        (u - 1) / w, 
        u / w))

    lam = K.concatenate((argmaxes, inflections))

    x_star_candidates = K.clip(u[:,None] - lam[:,:,None] * w[:,None], 0., 1.)

    # Evaluate `q` on all the candidate points for the max. Select the point
    # that is in fact the max (`opt_index`).
    max_q_candidates = (
        K.sum(
            .5 * (x_star_candidates - u[:,None])**2 + 
                lam[:,:,None] * w[:,None] * x_star_candidates,
            axis=2) - lam * b)

    opt_index = K.argmax(max_q_candidates, axis=1)

    # `x_star` is the projection satisfying the constraints.
    x_star = K.sum(
        K.one_hot(opt_index, 4*ndim - 1)[:,:,None] * x_star_candidates,
        axis=1)

    d = tf.norm(x_star - u, axis=1) / feasible
    
    return K.reshape(d, (-1, neurons))
