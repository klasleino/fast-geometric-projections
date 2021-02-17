import tensorflow.keras.backend as K
import numpy as np

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from fgp.projections import proj_dist
from fgp.utils import batch_flatten


class CheckableModel(object):

    def __init__(self, input_shape, layer_details, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.layer_details = layer_details

        # Construct the model.
        x = Input(self.input_shape, name='features')
        z = x

        # Keep track of the internal layer shapes.
        self.internal_layer_shapes = []
        self.hidden_layer_sizes = []

        flat = len(input_shape) == 1
        for layer_i_details in layer_details:
            if isinstance(layer_i_details, int) and not flat:
                # Flatten if we haven't already.
                z = Flatten()(z)
                flat = True
                
            z = self._layer_from_info(layer_i_details)(z)
            z = Activation('relu')(z)

            self.internal_layer_shapes.append(K.int_shape(z))
            self.hidden_layer_sizes.append(np.prod(K.int_shape(z)[1:]))

        self.logit = Dense(n_classes, activation='linear', name='logits')(z)
        y = Activation('softmax', name='probs')(self.logit)

        self.model = Model(x, y)

        self.output_activation_fn = K.function([x], [self.logit])

        self.internal_neuron_activation_fn = K.function(
            [x],
            [
                layer.output 
                for layer in self.model.layers[:-2] 
                if layer.get_weights()])

        # Flag to see if we've compiled the backprop functions.
        self.compiled = False
        
    def _layer_from_info(self, layer_i_details):
        if isinstance(layer_i_details, int):
            return Dense(layer_i_details, activation='linear')

        elif isinstance(layer_i_details, tuple) and (
                len(layer_i_details) == 2 or
                len(layer_i_details) == 3 or
                len(layer_i_details) == 4):

            # We interpret a length of 2 to be (num_kernels, kernel_size) and a
            # length of 3 to be (num_kernels, kernel_size, stride). If stride is
            # specified we do a "valid" convolution unless specified by the 4th
            # element of layer_i_details.
            return (
                Conv2D(
                    *layer_i_details, 
                    padding='same',
                    activation='linear') if len(layer_i_details) == 2 else
                Conv2D(
                    *layer_i_details[:2],
                    strides=layer_i_details[2],
                    padding='valid' if len(layer_i_details) == 3 else 
                        layer_i_details[3],
                    activation='linear'))

        else:
            raise ValueError('Unexpected format for `layer_details`')

    # Extend the functionality of self.model, e.g., compile, fit.
    def __getattr__(self, name):
        return getattr(self.model, name)

    def compile_backprop(self, proj_type='l2'):
        self.proj_type = proj_type
        
        layers = [
            layer for layer in self.model.layers if layer.get_weights()]
        
        A = [Input(layer.output_shape[1:]) for layer in layers[:-1]]

        # Store the constraint gradients and biases for each layer.
        grads, biases = [], []
        prev_grad, prev_bias = None, None
        
        for i, layer in enumerate(layers):
            W, b = layer.weights
            
            if isinstance(layer, Dense):
                if i > 0 and K.ndim(A[i-1]) == 4 and (
                        K.image_data_format() == 'channels_first'):

                    # The `Flatten` layer doesn't respect the channels-first
                    # dimension ordering, so it mixes up our dimensions. We need
                    # to correct for that here.
                    _, ch, h, w = K.int_shape(A[i-1])
                    _, n_out = K.int_shape(W)
                    W = K.reshape(
                        K.permute_dimensions(
                            K.reshape(W, (h,w,ch,n_out)),
                            (2,0,1,3)),
                        (ch*h*w, n_out))
                    
                if len(grads) == 0:
                    grad = K.transpose(W)
                    bias = b
                    
                    # Expand to batch shape.
                    grad = grad[None] * K.ones_like(A[i])[:,:,None]
                    bias = bias[None] * K.ones_like(A[i])
                        
                else:
                    A_i = K.reshape(
                        A[i-1], [-1, np.prod(K.int_shape(A[i-1])[1:])])
                    
                    grad = (K.transpose(W)[None] * A_i[:,None]) @ grads[-1]          
                    bias = (biases[-1] * A_i) @ W + b[None]
                    
                grads.append(grad)
                biases.append(bias)
                    
            else:
                if K.image_data_format() == 'channels_first':
                    _, ch_in, h_in, w_in = layer.input_shape
                    _, ch_out, h_out, w_out = layer.output_shape
                else:
                    _, h_in, w_in, ch_in = layer.input_shape
                    _, h_out, w_out, ch_out = layer.output_shape
                
                if len(grads) == 0:
                    if K.image_data_format() == 'channels_first':
                        grad = K.conv2d(
                            K.reshape(
                                K.eye(ch_in * h_in * w_in),
                                [ch_in * h_in * w_in, ch_in, h_in, w_in]),
                            W,
                            padding=layer.padding,
                            strides=layer.strides)

                        bias = K.tile(b[:,None,None], [1, h_out, w_out])

                    else:
                        grad = K.conv2d(
                            K.reshape(
                                K.eye(ch_in * h_in * w_in),
                                [ch_in * h_in * w_in, h_in, w_in, ch_in]),
                            W,
                            padding=layer.padding,
                            strides=layer.strides)

                        bias = K.tile(b[None,None], [h_out, w_out, 1])

                    # Expand to batch shape.
                    grad = grad[None] * K.ones_like(A[i])[:,None]
                    bias = bias[None] * K.ones_like(A[i])

                else:

                    n = np.prod(self.input_shape)

                    if K.image_data_format() == 'channels_first':
                        grad = K.reshape(
                            K.conv2d(
                                K.reshape(
                                    grad * A[i-1][:,None], 
                                    (-1, ch_in, h_in, w_in)),
                                W,
                                padding=layer.padding,
                                strides=layer.strides),
                            (-1, n, ch_out, h_out, w_out))

                        bias = K.conv2d(
                            bias * A[i-1],
                            W,
                            padding=layer.padding,
                            strides=layer.strides) + b[None,:,None,None]

                    else:
                        grad = K.reshape(
                            K.conv2d(
                                K.reshape(
                                    grad * A[i-1][:,None], 
                                    (-1, h_in, h_in, ch_in)),
                                W,
                                padding=layer.padding,
                                strides=layer.strides),
                            (-1, n, h_out, h_out, ch_out))

                        bias = K.conv2d(
                            bias * A[i-1],
                            W,
                            padding=layer.padding,
                            strides=layer.strides) + b[None,None,None]

                grads.append(
                    K.permute_dimensions(
                        K.reshape(grad, (-1, n, ch_out * h_out * w_out)),
                        (0,2,1)))
                biases.append(K.batch_flatten(bias))

        # Handle the softmax constraints.
        c = K.placeholder((1,), dtype='int32')

        softmax_grads = grads[-1]
        softmax_biases = biases[-1]

        c_grad = K.permute_dimensions(
            K.gather(K.permute_dimensions(softmax_grads, (1,0,2)), c),
            (1,0,2))

        c_bias = K.transpose(K.gather(K.transpose(softmax_biases), c))

        grads[-1] = softmax_grads - c_grad
        biases[-1] = softmax_biases - c_bias

        grads_no_first_layer = K.concatenate(grads[1:], axis=1)
        biases_no_first_layer = K.concatenate(biases[1:], axis=1)

        grads = K.concatenate(grads, axis=1)
        biases = K.concatenate(biases, axis=1)

        # Calculate distances.
        x = K.placeholder(self.input_shape)

        distances = proj_dist(proj_type, K.reshape(x, (1,-1)), grads, biases)

        distances_no_first_layer = proj_dist(
            proj_type, 
            K.reshape(x, (1,-1)), 
            grads_no_first_layer, 
            biases_no_first_layer)

        self._grad_f = K.function(A + [c], [grads])
        self._bias_f = K.function(A + [c], [biases])
        self._dist_f = K.function(A + [c, x], [distances])
        self._all_f = K.function(
            A + [c, x], 
            [distances, grads[:,-self.n_classes:], biases[:,-self.n_classes:]])

        self._all_except_first_f = K.function(
            A + [c, x],
            [
                distances_no_first_layer, 
                grads_no_first_layer[:,-self.n_classes:],
                biases_no_first_layer[:,-self.n_classes:]])

        self.compiled = True

        return self

    def get_internal_neuron_activation_pattern(self, x_0):
        if x_0.shape == self.input_shape:
            x_0 = np.expand_dims(x_0, axis=0)

        return np.concatenate(
            [
                batch_flatten((z > 0).astype('float32'))
                for z in self.internal_neuron_activation_fn([x_0])],
            axis=-1)

    def _unflatten_activation_pattern(self, pattern):
        unflattened, index = [], 0
        for shape in self.internal_layer_shapes:
            neurons = np.prod(shape[1:])
            unflattened.append(
                pattern[:,index:index+neurons].reshape(-1,*shape[1:]))

            index += neurons

        return unflattened

    def bprop_all(self, activations, x_0, c, use_cached=False):
        '''
        Optimization that allows us to compute the distances to activation 
        constraints directly on the GPU rather than returning the constraints 
        and calculating their distances on the CPU. For the output constraints, 
        the constraint itself returned, just as in `bprop_constraints`.
        '''
        if not self.compiled:
            raise AssertionError(
                'Must call `compile_backprop` before using `bprop_distances`')

        if len(activations.shape) is 1:
            activations = activations[None]

        activations = self._unflatten_activation_pattern(activations)

        if use_cached:
            top_distances, top_grads, top_biases = self._all_except_first_f(
                activations + [c, x_0])

            n = top_distances.shape[0]

            distances = np.concatenate(
                (np.tile(self.cached_first_layer[0], (n,1)), top_distances),
                axis=1)
            grads = np.concatenate(
                (np.tile(self.cached_first_layer[1], (n,1,1)), top_grads),
                axis=1)
            biases = np.concatenate(
                (np.tile(self.cached_first_layer[2], (n,1)), top_biases),
                axis=1)

        else:
            distances, grads, biases = self._all_f(activations + [c, x_0])

            self.cached_first_layer = [
                distances[0:1, :self.hidden_layer_sizes[0]],
                grads[0:1, :self.hidden_layer_sizes[0]],
                biases[0:1, :self.hidden_layer_sizes[0]]]

        return [distances, grads, biases]
