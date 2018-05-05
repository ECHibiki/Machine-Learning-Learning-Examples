Constants, Sequences, and Random Values

Several operations generate _constants_
tf.zeros( shape, dtype=tf.float32, name=None) 									// Creates Tensor of shape set to zero
tf.zeros_like( tensor,  dtype=None,  name=None,  optimize=True) 				// Creates a zero tensor based on another
tf.ones( shape, dtype=tf.float32,  name=None) 									// Creates Tensor of shape set to one
tf.ones_like(tensor, dtype=None, name=None, optimize=True) 						// Creates one tensor based on anotehr
tf.fill( dims, value, name=None) 												// Creates a tensor set to value
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False) 	// Generates a constant tensor	
																					Populated by type dtype. Value is either a set of values to create or a value followed by shape which is filled 
																			# Constant 1-D Tensor populated with value list.
																			tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

																			# Constant 2-D tensor populated with scalar value -1.
																			tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
																														 [-1. -1. -1.]]
_Sequences_
tf.linspace(start, stop, num,  name=None)	Also tf.lin_space		// Generates values in an interval :  tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
tf.range(start, limit, delta=1, dtype=None, name='range') 			//  Generates sequence of numbers in a range
																start = 3
																limit = 18
																delta = 3
																tf.range(start, limit, delta)  # [3, 6, 9, 12, 15]

The generation of random tensors with diferent distributions

		# Create a tensor of shape [2, 3] consisting of random normal values, with mean
		# -1 and standard deviation 4.
		norm = tf.random_normal([2, 3], mean=-1, stddev=4)

		# Shuffle the first dimension of a tensor
		c = tf.constant([[1, 2], [3, 4], [5, 6]])
		shuff = tf.random_shuffle(c)

		# Each time we run these ops, different results are generated
		sess = tf.Session()
		print(sess.run(norm))
		print(sess.run(norm))

		# Set an op-level seed to generate repeatable sequences across sessions.
		norm = tf.random_normal([2, 3], seed=1234)
		sess = tf.Session()
		print(sess.run(norm))
		print(sess.run(norm))
		sess = tf.Session()
		print(sess.run(norm))
		print(sess.run(norm))
		
		# Use random uniform values in [0, 1) as the initializer for a variable of shape
		# [2, 3]. The default type is float32.
		var = tf.Variable(tf.random_uniform([2, 3]), name="var")
		init = tf.global_variables_initializer()

		sess = tf.Session()
		sess.run(init)
		print(sess.run(var))
		
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) 		// Random numbers in a distribution
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)	// Randoms from a truncated normal
tf.random_uniform( shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)	// Randoms from a uniform distribution
tf.random_shuffle(value, seed=None, name=None)												// Randomly shuffles along a values first dimension
																									[[1, 2],       [[5, 6],
																									 [3, 4],  ==>   [1, 2],
																									 [5, 6]]        [3, 4]] The same pairs in different places
tf.random_crop(value, size, seed=None, name=None)											// A random slice of an array
tf.multinomial(logits, num_samples, seed=None, name=None, output_dtype=None)				// Random samples from a multinomial distribution
																									# samples has shape [1, 5], where each value is either 0 or 1 with equal
																									# probability.
																									samples = tf.multinomial(tf.log([[10., 10.]]), 5)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)			//Draws samples from a gamma distribution
tf.set_random_seed(seed) Sets graph-level random seed for random operations rellying on seed.