Tensor data can be casted with some of the following operations:
* tf.string_to_number(    string_tensor,    out_type=tf.float32,    name=None)	//Converts each string in the input Tensor to the specified numeric type.
* tf.to_double(    x,    name='ToDouble')		//Casts a tensor to type float64.
* tf.to_float(    x,    name='ToFloat')			//Casts a tensor to type float32.
* tf.to_bfloat16(   x,   name='ToBFloat16')		//Casts a tensor to type bfloat16.
* tf.to_int32(   x,    name='ToInt32')			//Casts a tensor to type int32.
* tf.to_int64(   x,  name='ToInt64')			//Casts a tensor to type int64.
* tf.cast(    x,    dtype,    name=None) 	 	//tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
* tf.bitcast(    input,    type,    name=None)	//Bitcasts a tensor from one type to another without copying data.
* tf.saturate_cast( value, dtype, name=None) 	//This function casts the input to dtype without applying any scaling

Tensor data can be changed in shape via the following
* tf.broadcast_dynamic_shape( shape_x,  shape_y) 			//tf.broadcast_dynamic_shape(A rank 1 integer Tensor, representing the shape of x: shape_x, A rank 1 integer Tensor, representing the shape of Y: shape_y) --Returns the broadcasted dynamic shape between shape_x and shape_y. 
* tf.broadcast_static_shape( shape_x,  shape_y) 			//Returns the broadcasted static shape between shape_x and shape_y.
* tf.shape(  input,   name=None,   out_type=tf.int32)		//tf.shape(input,name=None,out_type=tf.int32) t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]) => tf.shape(t) returns:   # [2, 2, 3]
* tf.shape_n( input,  out_type=tf.int32,  name=None)		//Returns shape of tensors.
* tf.size(  input,  name=None,  out_type=tf.int32)			//tf.size(input,name=None,out_type=tf.int32) t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]) => tf.size(t) returns:  # 12
* tf.rank(  input,  name=None)								//t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]) => tf.rank(t) returns:  # 3
* tf.reshape( tensor, shape, name=None)						//Given tensor, this operation returns a tensor that has the same values as tensor with shape shape. 

			# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
			# tensor 't' has shape [9]
			reshape(t, [3, 3]) ==> [[1, 2, 3],
									[4, 5, 6],
									[7, 8, 9]]

			# tensor 't' is [[[1, 1], [2, 2]],
			#                [[3, 3], [4, 4]]]
			# tensor 't' has shape [2, 2, 2]
			reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
									[3, 3, 4, 4]]

			# tensor 't' is [[[1, 1, 1],
			#                 [2, 2, 2]],
			#                [[3, 3, 3],
			#                 [4, 4, 4]],
			#                [[5, 5, 5],
			#                 [6, 6, 6]]]
			# tensor 't' has shape [3, 2, 3]
			# pass '[-1]' to flatten 't'
			reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

			# -1 can also be used to infer the shape

			# -1 is inferred to be 9:
			reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
									 [4, 4, 4, 5, 5, 5, 6, 6, 6]]
			# -1 is inferred to be 2:
			reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
									 [4, 4, 4, 5, 5, 5, 6, 6, 6]]
			# -1 is inferred to be 3:
			reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
										  [2, 2, 2],
										  [3, 3, 3]],
										 [[4, 4, 4],
										  [5, 5, 5],
										  [6, 6, 6]]]

			# tensor 't' is [7]
			# shape `[]` reshapes to a scalar
			reshape(t, []) ==> 7

tf.squeeze( input,  axis=None,  name=None, squeeze_dims=None) 				//Removes dimensions of size 1 from the shape of a tensor. # 't' is a tensor of shape [1, 2, 1, 3, 1, 1] => tf.shape(tf.squeeze(t)) returns: # [2, 3]
											Or
											# 't' is a tensor of shape [1, 2, 1, 3, 1, 1] => tf.shape(tf.squeeze(t, [2, 4])) returns:  # [1, 2, 3, 1]
tf.expand_dims( input,  axis=None,  name=None,  dim=None)					//Inserts a dimension of 1 into a tensor's shape. # 't' is a tensor of shape [2] => tf.shape(tf.expand_dims(t, 0)) returns: # [1, 2]
tf.meshgrid(*args, **kwargs) x = [1, 2, 3] y = [4, 5, 6] => X, Y = tf.meshgrid(x, y) # X = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] # Y = [[4, 4, 4], [5, 5, 5],  [6, 6, 6]]

Several operations to slice and join tensors
* tf.slice(input_,begin,size,name=None) // [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]] => tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
* tf.strided_slice( input_,  begin,  end,  strides=None,  begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, var=None,  name=None)		//t = tf.constant([[[1, 1, 1], [2, 2, 2]],
													 [[3, 3, 3], [4, 4, 4]],
													 [[5, 5, 5], [6, 6, 6]]])
								tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])  # [[[3, 3, 3]]]
								tf.strided_slice(t, [1, 0, 0], [2, 2, 3], [1, 1, 1])  # [[[3, 3, 3],
																					  #   [4, 4, 4]]]
								tf.strided_slice(t, [1, -1, 0], [2, -3, 3], [1, -1, 1])  # [[[4, 4, 4],
																						 #   [3, 3, 3]]]
								Instead of calling this op directly most users will want to use the NumPy-style slicing syntax (e.g. tensor[..., 3:4:-1, tf.newaxis, 3]),
								which is supported via tf.Tensor.getitem and tf.Variable.getitem
* tf.split( value,  num_or_size_splits,  axis=0,  num=None,  name='split')		//Splits a tensor into sub tensors.
							# 'value' is a tensor with shape [5, 30]
							# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
							split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
							tf.shape(split0)  # [5, 4]
							tf.shape(split1)  # [5, 15]
							tf.shape(split2)  # [5, 11]
							# Split 'value' into 3 tensors along dimension 1
							split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
							tf.shape(split0)  # [5, 10]
* tf.tile(input,multiples,name=None) //	creates a new tensor by replicating input multiples times. 
								 tiling [a b c d] by [2] produces [a b c d a b c d].
* tf.pad(tensor,paddings,mode='CONSTANT', name=None,constant_values=0)  //pads a tensor according to the paddings you specify.
						t = tf.constant([[1, 2, 3], [4, 5, 6]])
						paddings = tf.constant([[1, 1,], [2, 2]])
						# 'constant_values' is 0.
						# rank of 't' is 2.
						tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
														 #  [0, 0, 1, 2, 3, 0, 0],
														 #  [0, 0, 4, 5, 6, 0, 0],
														 #  [0, 0, 0, 0, 0, 0, 0]]

						tf.pad(t, paddings, "REFLECT")  # [[6, 5, 4, 5, 6, 5, 4],
														#  [3, 2, 1, 2, 3, 2, 1],
														#  [6, 5, 4, 5, 6, 5, 4],
														#  [3, 2, 1, 2, 3, 2, 1]]

						tf.pad(t, paddings, "SYMMETRIC")  # [[2, 1, 1, 2, 3, 3, 2],
														  #  [2, 1, 1, 2, 3, 3, 2],
														  #  [5, 4, 4, 5, 6, 6, 5],
														  #  [5, 4, 4, 5, 6, 6, 5]]
* tf.concat(values,axis,name='concat')				//Concatenates the list of tensors values along dimension axis.
												t1 = [[1, 2, 3], [4, 5, 6]]
												t2 = [[7, 8, 9], [10, 11, 12]]
												tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
												tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

												# tensor t3 with shape [2, 3]
												# tensor t4 with shape [2, 3]
												tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
												tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
* tf.stack(values, axis=0, name='stack')		//Packs the list of tensors in values into a tensor with rank one higher than each tensor in values, by packing them along the axis dimension. Given a list of length N of tensors of shape (A, B, C);
								x = tf.constant([1, 4])
								y = tf.constant([2, 5])
								z = tf.constant([3, 6])
								tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
								tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
								This is the opposite of unstack. The numpy equivalent is

								tf.stack([x, y, z]) = np.stack([x, y, z])
* tf.parallel_stack( values, name='parallel_stack')		//Stacks a list of rank-R tensors into one rank-(R+1) tensor in parallel.
								x = tf.constant([1, 4])
								y = tf.constant([2, 5])
								z = tf.constant([3, 6])
								tf.parallel_stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]]
* tf.unstack(value, num=None,  axis=0,  name='unstack') 	//Unpacks num tensors from value by chipping it along the axis dimension.
				For example, given a tensor of shape (A, B, C, D);
				If axis == 0 then the i'th tensor in output is the slice value[i, :, :, :] and each tensor in output will have shape (B, C, D). (Note that the dimension unpacked along is gone, unlike split).
				If axis == 1 then the i'th tensor in output is the slice value[:, i, :, :] and each tensor in output will have shape (A, C, D). Etc.
				tf.unstack(x, n) = np.unstack(x)
* tf.reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None, name=None, seq_dim=None, batch_dim=None)  //This op first slices input along the dimension batch_axis, and for each slice i, reverses the first seq_lengths[i] elements along the dimension seq_axis.
* tf.reverse(tensor,axis,name=None) 
				# tensor 't' is [[[[ 0,  1,  2,  3],
				#                  [ 4,  5,  6,  7],
				#                  [ 8,  9, 10, 11]],
				#                 [[12, 13, 14, 15],
				#                  [16, 17, 18, 19],
				#                  [20, 21, 22, 23]]]]
				# tensor 't' shape is [1, 2, 3, 4]

				# 'dims' is [3] or 'dims' is [-1]
				reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
										[ 7,  6,  5,  4],
										[ 11, 10, 9, 8]],
									   [[15, 14, 13, 12],
										[19, 18, 17, 16],
										[23, 22, 21, 20]]]]

				# 'dims' is '[1]' (or 'dims' is '[-3]')
				reverse(t, dims) ==> [[[[12, 13, 14, 15],
										[16, 17, 18, 19],
										[20, 21, 22, 23]
									   [[ 0,  1,  2,  3],
										[ 4,  5,  6,  7],
										[ 8,  9, 10, 11]]]]

				# 'dims' is '[2]' (or 'dims' is '[-2]')
				reverse(t, dims) ==> [[[[8, 9, 10, 11],
										[4, 5, 6, 7],
										[0, 1, 2, 3]]
									   [[20, 21, 22, 23],
										[16, 17, 18, 19],
										[12, 13, 14, 15]]]]
* tf.reverse_v2 		//NOTE tf.reverse has now changed behavior in preparation for 1.0. tf.reverse_v2 is currently an alias that will be deprecated before TF 1.0.
* tf.transpose(a,perm=None,name='transpose',conjugate=False) 	//Transposes a. Permutes the dimensions according to perm.

			x = tf.constant([[1, 2, 3], [4, 5, 6]])
			tf.transpose(x)  # [[1, 4]
							 #  [2, 5]
							 #  [3, 6]]

			# Equivalently
			tf.transpose(x, perm=[1, 0])  # [[1, 4]
										  #  [2, 5]
										  #  [3, 6]]

			# If x is complex, setting conjugate=True gives the conjugate transpose
			x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
							 [4 + 4j, 5 + 5j, 6 + 6j]])
			tf.transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
											 #  [2 - 2j, 5 - 5j],
											 #  [3 - 3j, 6 - 6j]]

			# 'perm' is more useful for n-dimensional tensors, for n > 2
			x = tf.constant([[[ 1,  2,  3],
							  [ 4,  5,  6]],
							 [[ 7,  8,  9],
							  [10, 11, 12]]])

			# Take the transpose of the matrices in dimension-0
			# (this common operation has a shorthand `matrix_transpose`)
			tf.transpose(x, perm=[0, 2, 1])  # [[[1,  4],
											 #   [2,  5],
											 #   [3,  6]],
											 #  [[7, 10],
											 #   [8, 11],
											 #   [9, 12]]]
* tf.extract_image_patches(Tensor images, intlist  ksizes, strides, rates, padding, name=None)			//Extract patches from images and put them in the "depth" output dimension
* tf.space_to_batch_nd(input, block_shape, paddings, name=None)			// divides "spatial" dimensions [1, ..., M] of the input into a grid of blocks of shape block_shape, and interleaves these blocks with the "batch" dimension (0) such that in the output, the spatial dimensions [1, ..., M] correspond to the position within the grid, and the batch dimension combines both the position within a spatial block and the original batch position.
* tf.space_to_batch(4DTensor input,  paddings,  block_size,  name=None)			//Zero-pads and then rearranges (permutes) blocks of spatial data into batch. More specifically, this op outputs a copy of the input tensor where values from the height and width dimensions are moved to the batch dimension. 
* tf.required_space_to_batch_paddings( input_shape, block_shape, base_paddings=None, name=None)		//This function can be used to calculate a suitable paddings argument for use with space_to_batch_nd and batch_to_space_nd.
* tf.batch_to_space_nd(input, block_shape, crops, name=None)					//This operation reshapes the "batch" dimension 0 into M + 1 dimensions of shape block_shape + [batch], interleaves these blocks back into the grid defined by the spatial dimensions [1, ..., M], to obtain a result with the same rank as the input.
* tf.batch_to_space 			//This is a legacy version of the more general BatchToSpaceND.
* tf.space_to_depth(    input,    block_size,    name=None,    data_format='NHWC')
					Shape [1, 2, 2, 1], data_format = "NHWC" and block_size = 2:
					x = [[[[1], [2]],
						  [[3], [4]]]]
					This operation will output a tensor of shape [1, 1, 1, 4]:
					[[[[1, 2, 3, 4]]]]			
* tf.depth_to_space( input,  block_size,  name=None,  data_format='NHWC')
					 shape [1, 1, 1, 4], data_format = "NHWC" and block_size = 2:

					x = [[[[1, 2, 3, 4]]]]

					This operation will output a tensor of shape [1, 2, 2, 1]:

					   [[[[1], [2]],
						 [[3], [4]]]]
* tf.gather(params, indices, validate_indices=None, name=None, axis=0)
	slices from params axis axis according to indices.
* tf.gather_nd(    params,    indices,    name=None)
	Gather slices from params into a Tensor with shape specified by indices. indices defines slices into the first N dimensions of params, where N = indices.shape[-1].
			Simple indexing into a matrix:

			indices = [[0, 0], [1, 1]]
			params = [['a', 'b'], ['c', 'd']]
			output = ['a', 'd']
		Slice indexing into a matrix:

			indices = [[1], [0]]
			params = [['a', 'b'], ['c', 'd']]
			output = [['c', 'd'], ['a', 'b']]
		Indexing into a 3-tensor:

			indices = [[1]]
			params = [[['a0', 'b0'], ['c0', 'd0']],
					  [['a1', 'b1'], ['c1', 'd1']]]
			output = [[['a1', 'b1'], ['c1', 'd1']]]

			indices = [[0, 1], [1, 0]]
			params = [[['a0', 'b0'], ['c0', 'd0']],
					  [['a1', 'b1'], ['c1', 'd1']]]
			output = [['c0', 'd0'], ['a1', 'b1']]

			indices = [[0, 0, 1], [1, 0, 1]]
			params = [[['a0', 'b0'], ['c0', 'd0']],
					  [['a1', 'b1'], ['c1', 'd1']]]
			output = ['b0', 'b1']
* tf.unique_with_counts(x, out_idx=tf.int32, name=None)
	This operation returns a tensor y containing all of the unique elements of x sorted in the same order that they occur in x.
		# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
		y, idx, count = unique_with_counts(x)
		y ==> [1, 2, 4, 7, 8]
		idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
		count ==> [2, 1, 3, 1, 2]
* tf.scatter_nd( indices,  updates,  shape,  name=None)	//Creates a new tensor by applying sparse updates to individual values or slices within a zero tensor of the given shape according to indices. This operator is the inverse of the tf.gather_nd operator 
* tf.dynamic_partition( data,  partitions,  num_partitions,  name=None)	//Partitions data into num_partitions tensors using indices from partitions.
				# Scalar partitions.
				partitions = 1
				num_partitions = 2
				data = [10, 20]
				outputs[0] = []  # Empty with shape [0, 2]
				outputs[1] = [[10, 20]]

				# Vector partitions.
				partitions = [0, 0, 1, 1, 0]
				num_partitions = 2
				data = [10, 20, 30, 40, 50]
				outputs[0] = [10, 20, 50]
				outputs[1] = [30, 40]
* tf.dynamic_stitch(    data,    partitions,    num_partitions,    name=None)	//Interleave the values from the data tensors into a single tensor.
		indices[0] = 6
		indices[1] = [4, 1]
		indices[2] = [[5, 2], [0, 3]]
		data[0] = [61, 62]
		data[1] = [[41, 42], [11, 12]]
		data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
		merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
				  [51, 52], [61, 62]]
* tf.boolean_mask(    tensor,    mask,    name='boolean_mask',    axis=None) //Apply boolean mask to tensor. Numpy equivalent is tensor[mask].
	# 2-D example
	tensor = [[1, 2], [3, 4], [5, 6]]
	mask = np.array([True, False, True])
	boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
* tf.one_hot(    indices,    depth,    on_value=None,    off_value=None,    axis=None,    dtype=None,    name=None)//The locations represented by indices in indices take value on_value, while all other locations take value off_value.
			indices = [0, 1, 2]
			depth = 3
			tf.one_hot(indices, depth) 
			# output: [3 x 3]
			# [[1., 0., 0.],
			#  [0., 1., 0.],
			#  [0., 0., 1.]]

			indices = [0, 2, -1, 1]
			depth = 3
			tf.one_hot(indices, depth,
					   on_value=5.0, off_value=0.0,
					   axis=-1)  # output: [4 x 3]
			# [[5.0, 0.0, 0.0],  # one_hot(0)
			#  [0.0, 0.0, 5.0],  # one_hot(2)
			#  [0.0, 0.0, 0.0],  # one_hot(-1)
			#  [0.0, 5.0, 0.0]]  # one_hot(1)

			indices = [[0, 2], [1, -1]]
			depth = 3
			tf.one_hot(indices, depth,
					   on_value=1.0, off_value=0.0,
					   axis=-1)  # output: [2 x 2 x 3]
			# [[[1.0, 0.0, 0.0],   # one_hot(0)
			#   [0.0, 0.0, 1.0]],  # one_hot(2)
			#  [[0.0, 1.0, 0.0],   # one_hot(1)
			#   [0.0, 0.0, 0.0]]]  # one_hot(-1)
* tf.sequence_mask( lengths,  maxlen=None,  dtype=tf.bool,  name=None)		//Returns a mask tensor representing the first N positions of each cell.

		tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                #  [True, True, True, False, False],
                                #  [True, True, False, False, False]]

		tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
										  #   [True, True, True]],
										  #  [[True, True, False],
										  #   [False, False, False]]]
										  
* tf.dequantize(  input,  min_range,  max_range,  mode='MIN_COMBINED',  name=None)
	//used to convert the float values to their quantized equivalents.
* tf.quantize(    input,    min_range,    max_range,    T,    mode='MIN_COMBINED',    round_mode='HALF_AWAY_FROM_ZERO',    name=None)
	//Quantization is the process of constraining an input from a continuous or otherwise large set of values (such as the real numbers) to a discrete set (such as the integers). 
	//The terms quantization and discretization are often denotatively synonymous but not always connotatively interchangeable.
* tf.quantize_v2			//Please use tf.quantize instead.
* tf.quantized_concat(   concat_dim,   values,   input_mins,   input_maxes,   name=None)		//Concatenates quantized tensors along one dimension.
* tf.setdiff1d(  x,   y,   index_dtype=tf.int32,   name=None) 		//Given a list x and a list y, this operation returns a list out that represents all values that are in x but not in y
		For example, given this input:
		x = [1, 2, 3, 4, 5, 6]
		y = [1, 3, 5]
		This operation would return:

		out ==> [2, 4, 6]
		idx ==> [1, 3, 5]

Operations used to help train for better quantization accuracy.

tf.fake_quant_with_min_max_args( inputs,  min=-6,  max=6,  num_bits=8,  narrow_range=False,  name=None) //Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type. 
			//Quantization is called fake since the output is still in floating point.
tf.fake_quant_with_min_max_args_gradient( gradients, inputs, min=-6, max=6, num_bits=8, narrow_range=False, name=None) //Compute gradients for a FakeQuantWithMinMaxArgs operation.
tf.fake_quant_with_min_max_vars(inputs, min, max, num_bits=8, narrow_range=False, name=None)   //Fake-quantize the 'inputs' tensor of type float via global float scalars min and max to 'outputs' tensor of same shape as inputs. [min; max] define the clamping range for the inputs data.
tf.fake_quant_with_min_max_vars_gradient( gradients, inputs, min, max, num_bits=8, narrow_range=False,  name=None) //Compute gradients for a FakeQuantWithMinMaxVars operation.
tf.fake_quant_with_min_max_vars_per_channel( inputs,  min,  max,  num_bits=8,  narrow_range=False,  name=None)       //Fake-quantize the 'inputs' tensor of type float and one of the shapes: [d],[b, d] [b, h, w, d] via per-channel floats min and max of shape [d] to 'outputs' tensor of same shape as inputs.
tf.fake_quant_with_min_max_vars_per_channel_gradient( gradients, inputs, min, max, num_bits=8, narrow_range=False, name=None) 		//Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.