from keras.layers import Activation, Dense, Input
from keras.models import Model

import keras.backend as K
import numpy as np


K_l2_norm = lambda x: K.sqrt(K.sum(x * x, axis=-1))

class SimpleSigmoidFfn(object):
	'''
	Implements a fully-connected Keras model with specified layer sizes that 
	outputs a single sigmoid prediction for binary classification tasks. Extends 
	the functionality of the Keras model to allow the efficient computation of 
	activations and constraints.
	'''
	def __init__(self, input_size, hidden_layer_sizes):
		self.input_shape = (input_size,)

		self.hidden_layer_sizes = hidden_layer_sizes

		# Construct the model.
		x = Input(self.input_shape, name='features')
		z = x
		for size in hidden_layer_sizes:
			z = Dense(size, activation='linear')(z)
			z = Activation('relu')(z)
		self.logit = Dense(1, activation='linear', name='logits')(z)
		y = Activation('sigmoid', name='probs')(self.logit)

		self.model = Model(x, y)

		self.output_activation_fn = K.function([x], [K.sum(self.logit)])

		# Compile the activation function.
		dense_layer_indices = [2*i + 1 for i in range(len(self.hidden_layer_sizes))]
		self.internal_neuron_activation_fn = K.function(
			[x],
			[
				self.model.layers[l].output[:,j]
				for i, l in enumerate(dense_layer_indices)
				for j in range(self.hidden_layer_sizes[i])])

		# Flag to see if we've compiled the backprop functions.
		self.compiled = False

	# Extend the functionality of self.model, e.g., compile, fit.
	def __getattr__(self, name):
		return getattr(self.model, name)

	def compile_backprop(self):
		y = K.placeholder((None,1))

		hidden_layer_indices = np.concatenate((
			[0], np.cumsum(self.hidden_layer_sizes))).astype('int32')

		layers = len(self.hidden_layer_sizes)

		# Gives the activation pattern we are calculating the gradient for.
		activations = K.placeholder((None,hidden_layer_indices[-1]))

		non_input_layer_sizes = np.concatenate((
			self.hidden_layer_sizes, [1])).astype('int32')
		grads = []
		for start in range(layers, -1, -1):

			# Calculate the gradient w.r.t. input for each neuron in the start layer.
			for j in range(non_input_layer_sizes[start]-1, -1, -1):
				z = K.dot(y, K.transpose(self.model.weights[2*start])[j:j+1])

				if start is not 0:
					for i in range(start - 1, -1, -1):
						# Mask according to the given activation pattern.
						z = z * activations[
							:, hidden_layer_indices[i]:hidden_layer_indices[i+1]]

						z = K.dot(z, K.transpose(self.model.weights[2*i]))
						
				grads.append(z)

		# We stored the gradients in the reverse order from how they are computed
		# with Keras, so reverse them.
		grads.reverse()

		self.backprop_gradient = K.function([y, activations], grads)

		# Now create a function that computes the bias from the activation pattern.
		x = K.placeholder((None,) + self.input_shape)
		z = x

		biases = []
		for i in range(layers + 1):
			z = K.dot(z, self.model.weights[2*i]) + self.model.weights[2*i + 1]

			for j in range((tuple(self.hidden_layer_sizes) + (1,))[i]):
				biases.append(z[:,j])

			if i < layers:
				z *= activations[:, hidden_layer_indices[i]:hidden_layer_indices[i+1]]

		self.get_biases = K.function([x, activations], biases)

		self.compiled = True

		return self

	def get_internal_neuron_activation_pattern(self, x_0):
		if x_0.shape == self.input_shape:
			x_0 = np.expand_dims(x_0, axis=0)
		return (
			(np.transpose(self.internal_neuron_activation_fn([x_0])) > 0)
			.astype('float32'))

	def bprop_cosntraints(self, activations, c):
		if not self.compiled:
			raise AssertionError(
				'Must call `compile_backprop` before using `bprop_cosntraints`')

		if len(activations.shape) is 1:
			activations = np.expand_dims(activations, axis=0)

		n = activations.shape[0]

		# Keep track of the direction each constraint faces so we can check if the
		# constraint is satisfied when using an LP solver.
		sign = np.expand_dims(
			np.concatenate((
					activations * -2 + 1, 
					np.ones((n,1)) * (c * -2 + 1)),
				axis=-1).transpose(),
			axis=-1)

		return zip(
			sign * self.backprop_gradient([np.ones((n,1)), activations]),
			sign[:,:,0] * 
				self.get_biases([np.zeros((n,) + self.input_shape), activations]))

	def bprop_distances(self, activations, x_0, c):
		raise NotImplementedError(
			'This optimization is currently only implemented for softmax networks.')


class SimpleSoftmaxFfn(object):
	'''
	Implements a fully-connected Keras model with specified layer sizes that 
	outputs a single sigmoid prediction for binary classification tasks. Extends 
	the functionality of the Keras model to allow the efficient computation of 
	activations and constraints.
	'''
	def __init__(self, input_size, hidden_layer_sizes, n_classes):
		self.input_shape = (input_size,)
		self.n_classes = n_classes
		self.hidden_layer_sizes = hidden_layer_sizes

		# Construct the model.
		x = Input(self.input_shape, name='features')
		z = x
		for size in hidden_layer_sizes:
			z = Dense(size, activation='linear')(z)
			z = Activation('relu')(z)
		self.logit = Dense(n_classes, activation='linear', name='logits')(z)
		y = Activation('softmax', name='probs')(self.logit)

		self.model = Model(x, y)

		self.output_activation_fn = K.function([x], [self.logit])

		dense_layer_indices = [2*i + 1 for i in range(len(self.hidden_layer_sizes))]
		self.internal_neuron_activation_fn = K.function(
			[x],
			[
				self.model.layers[l].output[:,j]
				for i, l in enumerate(dense_layer_indices)
				for j in range(self.hidden_layer_sizes[i])])

		# Flag to see if we've compiled the backprop functions.
		self.compiled = False

	# Extend the functionality of self.model, e.g., compile, fit.
	def __getattr__(self, name):
		return getattr(self.model, name)

	def compile_backprop(self):
		y = K.placeholder((None,1))
		x_0 = K.placeholder((1,) + self.input_shape)

		hidden_layer_indices = np.concatenate((
			[0], np.cumsum(self.hidden_layer_sizes))).astype('int32')

		layers = len(self.hidden_layer_sizes)

		# Gives the activation pattern we are calculating the gradient for.
		activations = K.placeholder((None,hidden_layer_indices[-1]))

		non_input_layer_sizes = np.concatenate((
			self.hidden_layer_sizes, [self.n_classes])).astype('int32')
		grads = []
		for start in range(layers, -1, -1):

			# Calculate the gradient w.r.t. input for each neuron in the start layer.
			for j in range(non_input_layer_sizes[start]-1, -1, -1):
				z = K.dot(y, K.transpose(self.model.weights[2*start])[j:j+1])

				if start is not 0:
					for i in range(start - 1, -1, -1):
						# Mask according to the given activation pattern.
						z = z * activations[
							:, hidden_layer_indices[i]:hidden_layer_indices[i+1]]

						z = K.dot(z, K.transpose(self.model.weights[2*i]))
						
				grads.append(z)

		# We stored the gradients in the reverse order from how they are computed
		# with Keras, so reverse them.
		grads.reverse()

		self.backprop_gradient = K.function([y, activations], grads)
		self.backprop_gradient_output = K.function(
			[y, activations], 
			grads[-self.n_classes:])

		# Now create a function that computes the bias from the activation pattern.
		x = K.placeholder((None,) + self.input_shape)
		z = x

		biases = []
		for i in range(layers + 1):
			z = K.dot(z, self.model.weights[2*i]) + self.model.weights[2*i + 1]

			for j in range((tuple(self.hidden_layer_sizes) + (self.n_classes,))[i]):
				biases.append(z[:,j])

			if i < layers:
				z *= activations[:, hidden_layer_indices[i]:hidden_layer_indices[i+1]]

		self.get_biases = K.function([x, activations], biases)
		self.get_biases_output = K.function(
			[x, activations], 
			biases[-self.n_classes:])

		# Compute the distances.
		distances = [
			K.abs(K.sum(x_0 * grad, axis=-1) + bias) /  K_l2_norm(grad)
			for grad, bias in zip(grads[:-self.n_classes], biases[:-self.n_classes])]

		self.get_distances = K.function([y, x, x_0, activations], distances)

		self.compiled = True

		return self

	def get_internal_neuron_activation_pattern(self, x_0):
		if x_0.shape == self.input_shape:
			x_0 = np.expand_dims(x_0, axis=0)
		return (
			(np.transpose(self.internal_neuron_activation_fn([x_0])) > 0)
			.astype('float32'))

	def bprop_cosntraints(self, activations, c):
		if not self.compiled:
			raise AssertionError(
				'Must call `compile_backprop` before using `bprop_cosntraints`')

		if len(activations.shape) is 1:
			activations = np.expand_dims(activations, axis=0)

		n = activations.shape[0]

		if isinstance(c, int):
			c = c * np.ones((n,1)).astype('int32')

		# Keep track of the direction each constraint faces so we can check if the
		# constraint is satisfied when using an LP solver.
		sign = np.expand_dims(
			np.concatenate((
					activations * -2 + 1, 
					np.ones((n, self.n_classes - 1))),
				axis=-1).transpose(),
			-1)

		grads = np.array(self.backprop_gradient([np.ones((n,1)), activations]))

		biases = np.array(
			self.get_biases([np.zeros((n,) + self.input_shape), activations]))

		c_compl = np.arange(self.n_classes)[None,:] != c

		# For the softmax constraints we need to subtract the constraint for class
		# `c` from each of the other output constraints to get constraints of the
		# form w_c * x + b_c >= w_c' * x _ b_c' for all c' != c.
		output_grads = grads[-self.n_classes:]
		softmax_grads = np.array([
			output_grads[c[i], i] - output_grads[others, i]
			for i, others in enumerate(c_compl)]).transpose(1,0,2)

		grads = np.concatenate((grads[:-self.n_classes], softmax_grads))

		output_biases = biases[-self.n_classes:]
		softmax_biases = np.array([
			output_biases[c[i], i] - output_biases[others, i]
			for i, others in enumerate(c_compl)]).transpose()

		biases = np.concatenate((biases[:-self.n_classes], softmax_biases))

		return zip(sign * grads, sign[:,:,0] * biases)

	def bprop_distances(self, activations, x_0, c):
		'''
		Optimization that allows us to compute the distances to activation 
		constraints directly on the GPU rather than returning the constraints and 
		calculating their distances on the CPU. For the output constraints, the 
		constraint itself returned, just as in `bprop_constraints`.
		'''
		if not self.compiled:
			raise AssertionError(
				'Must call `compile_backprop` before using `bprop_distances`')

		if len(activations.shape) is 1:
			activations = activations[None]

		if len(x_0.shape) is 1:
			x_0 = x_0[None]

		n = activations.shape[0]

		if isinstance(c, int):
			c = c * np.ones((n,1)).astype('int32')

		sign = np.ones((self.n_classes - 1, n, 1))

		distances = self.get_distances([
			np.ones((n,1)), np.zeros((n,) + self.input_shape), x_0, activations])

		output_grads = np.array(
			self.backprop_gradient_output([np.ones((n,1)), activations]))

		output_biases = np.array(
			self.get_biases_output([np.zeros((n,) + self.input_shape), activations]))

		c_compl = np.arange(self.n_classes)[None,:] != c

		# For the softmax constraints we need to subtract the constraint for class
		# `c` from each of the other output constraints to get constraints of the
		# form w_c * x + b_c >= w_c' * x _ b_c' for all c' != c.
		softmax_grads = np.array([
			output_grads[c[i], i] - output_grads[others, i]
			for i, others in enumerate(c_compl)]).transpose(1,0,2)

		softmax_biases = np.array([
			output_biases[c[i], i] - output_biases[others, i]
			for i, others in enumerate(c_compl)]).transpose()

		return (
			distances + list(zip(sign * softmax_grads, sign[:,:,0] * softmax_biases)))
