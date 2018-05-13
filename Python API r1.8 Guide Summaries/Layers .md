Layers

Ops for building neural network layers, regularizers, summaries, etc.

Higher level ops for building neural network layers
This package provides several ops that take care of creating variables that are used internally in a consistent way.
The building blocks for many common machine learning algorithms.
Aliases for fully_connected which set a default activation function are available: relu, relu6 and linear.

* tf.contrib.layers.avg_pool2d(inputs,kernel_size,stride=2,padding='VALID',data_format=DATA_FORMAT_NHWC,outputs_collections=None,scope=None)   // Adds a 2D average pooling operation done per image.
* tf.contrib.layers.batch_norm(inputs,decay=0.999,center=True,scale=False,epsilon=0.001,activation_fn=None,param_initializers=None,param_regularizers=None,updates_collections=tf.GraphKeys.UPDATE_OPS,is_training=True,reuse=None,variables_collections=None,outputs_collections=None,trainable=True,batch_weights=None,fused=None,data_format=DATA_FORMAT_NHWC,zero_debias_moving_mean=False,scope=None,renorm=False,renorm_clipping=None,renorm_decay=0.99,adjustment=None)
	// Adds batch normalization as defined by https://arxiv.org/abs/1502.03167 . Used as normalizer function for conv2D and fully_connected
* tf.contrib.layers.convolution2d(inputs,num_outputs,kernel_size,stride=1,padding='SAME',data_format=None,rate=1,activation_fn=tf.nn.relu,normalizer_fn=None,normalizer_params=None,weights_initializer=initializers.xavier_initializer(),weights_regularizer=None,biases_initializer=tf.zeros_initializer(),biases_regularizer=None,reuse=None,variables_collections=None,outputs_collections=None,trainable=True,scope=None)
	// Adds an N-D convoluition with an optional batch layer norm. Convolution creates a variable of weights representing the kernel.
* tf.contrib.layers.conv2d_in_plane(inputs,kernel_size,stride=1,padding='SAME',activation_fn=tf.nn.relu,normalizer_fn=None,normalizer_params=None,weights_initializer=initializers.xavier_initializer(),weights_regularizer=None,biases_initializer=tf.zeros_initializer(),biases_regularizer=None,reuse=None,variables_collections=None,outputs_collections=None,trainable=True,scope=None)
	// Performs convolutions to each channel independently. 
* tf.nn.conv2d_transpose(value,filter,output_shape,strides,padding='SAME',data_format='NHWC',name=None)
	// Sometimes called deconvolution after  http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf rather is the transpose gradient of conv2d
* tf.nn.dropout(x,keep_prob,noise_shape=None,seed=None,name=None)
	// General dropout function. Default is each element is kept or dropped independently.
* tf.contrib.layers.flatten(inputs,outputs_collections=None,scope=None)
	// Flattens but inputs batch size
* tf.contrib.layers.fully_connected(inputs,num_outputs,activation_fn=tf.nn.relu,normalizer_fn=None,normalizer_params=None,weights_initializer=initializers.xavier_initializer(),weights_regularizer=None,biases_initializer=tf.zeros_initializer(),biases_regularizer=None,reuse=None,variables_collections=None,outputs_collections=None,trainable=True,scope=None)
	// Adds a fully connected layer. Variable weights representing wiehgt matrix multiplied by inputs to form a tensor.
* tf.contrib.layers.layer_norm(inputs,center=True,scale=True,activation_fn=None,reuse=None,variables_collections=None,outputs_collections=None,trainable=True,begin_norm_axis=1,begin_params_axis=-1,scope=None)
	// Adds a normalization layer based on https://arxiv.org/abs/1607.06450 . Can be seen as normalizer for conv2d and fully_connected
* tf.contrib.layers.max_pool2d(inputs,kernel_size,stride=2,padding='VALID',data_format=DATA_FORMAT_NHWC,outputs_collections=None,scope=None)
	// 2D max pooling OP. Done per image not in batch or channels
* tf.contrib.layers.one_hot_encoding(labels,num_classes,on_value=1.0,off_value=0.0,outputs_collections=None,scope=None)
	// Transforms numeric labels to onehot_labels of type tf.one_hot
* tf.nn.relu(features,name=None)
	//The rectified linear max function max(features, 0)
* tf.nn.relu6(features,name=None)
	// Rectified linear max min 6 min(max(features, 0), 6).
* tf.contrib.layers.repeat(inputs,repetitions,layer,*args,**kwargs)
	// Repeats the same layer with the same arguments [y = repeat(x, 3, conv2d, 64, [3, 3], scope='conv1')]
* tf.contrib.layers.safe_embedding_lookup_sparse(embedding_weights,sparse_ids,sparse_weights=None,combiner=None,default_id=None,name=None,partition_strategy='div',max_norm=None)
	// Looks up embedding results. Invalid IDs are pruned. May be multidimensional
* tf.nn.separable_conv2d(input,depthwise_filter,pointwise_filter,strides,padding,rate=None,name=None,data_format=None)
	// conv2d where filters may seperate. Depthwise convolution on seperate channels follwed by pointwise mixing channels
* tf.contrib.layers.separable_convolution2d(inputs,num_outputs,kernel_size,depth_multiplier,stride=1,padding='SAME',data_format=DATA_FORMAT_NHWC,rate=1,activation_fn=tf.nn.relu,normalizer_fn=None,normalizer_params=None,weights_initializer=initializers.xavier_initializer(),weights_regularizer=None,biases_initializer=tf.zeros_initializer(),biases_regularizer=None,reuse=None,variables_collections=None,outputs_collections=None,trainable=True,scope=None)
	// Alias of above
* tf.nn.softmax DEPRECIATED [softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)]
* tf.stack(values,axis=0,name='stack')
	//Stacks a list of rank tensors into a rank+1 tensor
* tf.contrib.layers.unit_norm(inputs,dim,epsilon=1e-07,scope=None)
	// Normalizes input across dimension to unit length. Input of rank must be known.
* tf.contrib.layers.embed_sequence(ids,vocab_size=None,embed_dim=None,unique=False,initializer=None,regularizer=None,trainable=True,scope=None,reuse=None)
	//Maps sequence of symbols to a sequence of embeddings. For reusing embeddings between encoder and decoder.

Regularizers
Regularization can help prevent overfitting. These have the signature fn(weights). The loss is typically added to tf.GraphKeys.REGULARIZATION_LOSSES.

* tf.contrib.layers.apply_regularization(regularizer,weights_list=None) 
	// Gets summed penalty of regularizers to prevent overfitting
* tf.contrib.layers.l1_regularizer(scale,scope=None)
	// Gets the function for L1 regularization. L1 encourages sparsity
* tf.contrib.layers.l2_regularizer(scale,scope=None)
	// Function to apply L2 Regularization. 
* tf.contrib.layers.sum_regularizer(regularizer_list,scope=None)
	// Returns function for sum of multiple regularizers
Initializers
Initializers are used to initialize variables with sensible values given their size, data type, and purpose.

* tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)
	// Initializer of xiaver initialization of weights from http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
* tf.contrib.layers.xavier_initializer_conv2d
	// xiaver initialization for conv2D
* tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False,seed=None,dtype=tf.float32)
	// Generates tensors without scaling varience
		  if mode='FAN_IN': # Count only number of input connections.
			n = fan_in
		  elif mode='FAN_OUT': # Count only number of output connections.
			n = fan_out
		  elif mode='FAN_AVG': # Average number of inputs and output connections.
			n = (fan_in + fan_out)/2.0
			truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
Optimization
Optimize weights given a loss.
* tf.contrib.layers.optimize_loss(loss,global_step,learning_rate,optimizer,gradient_noise_scale=None,gradient_multipliers=None,clip_gradients=None,learning_rate_decay_fn=None,update_ops=None,variables=None,name=None,summaries=None,colocate_gradients_with_ops=False,increment_global_step=True)
	// With loss and paramters for optimizer, gives a training OP

Summaries
Helper functions to summarize specific variables or ops.
* tf.contrib.layers.summarize_activation(op) // Takes an Operation and adds useful summaries about it
* tf.contrib.layers.summarize_tensor(tensor,tag=None) // Scalar tensors produce a scalar_summary, for all others histogram_summary is produced
* tf.contrib.layers.summarize_tensors(tensors,summarizer=tf.contrib.layers.summarize_tensor) //Summarize a set of tensors using above methods
* tf.contrib.layers.summarize_collection(collection,name_filter=None,summarizer=tf.contrib.layers.summarize_tensor) //Summarize a graph collection of tensors, possibly filtered by name.


The layers module defines convenience functions summarize_variables, summarize_weights and summarize_biases, which set the collection argument of summarize_collection to VARIABLES, WEIGHTS and BIASES, respectively.
* tf.contrib.layers.summarize_activations(name_filter=None,summarizer=tf.contrib.layers.summarize_activation) // Uses summarize_activation  to sumarize activations
Feature columns
Feature columns provide a mechanism to map data to a model.
* tf.contrib.layers.bucketized_column(source_column,boundaries) //BucketizedColumn for discretizing dense input. https://stats.stackexchange.com/questions/266996/what-do-dense-and-sparse-mean-in-the-context-of-neural-networks
 Dense=='Sparse' and 'dense' refer to the number of zero vs. nonzero elements in an array (e.g. vector or matrix). A sparse array is one that contains mostly zeros, and few non-zero entries. A dense array contains mostly nonzeros. The usage of these terms in the context of neural nets is similar to their usage in other fields. There's no hard threshold for what counts as sparse; it's a loose term, but can be made more specific. For example, a vector is k-sparse if it contains at most k nonzero entries. Another way of saying this is that the vector's ℓ0 norm is k. Things that may be described as sparse or dense include the activations of units within a particular layer, the weights, and the data. One could also talk about 'sparse connectivity', which refers to the situation where only a small subset of units are connected to each other. This is a similar concept to sparse weights, because a connection with zero weight is effectively unconnected. 'Sparse array' can also refer to a class of data types that are efficient for representing arrays that are sparse. This is a concept within the domain of programming languages. It's related to, but distinct from the mathematical concept.
* tf.contrib.layers.check_feature_columns(feature_columns) //Checks the validity of the set of FeatureColumns.
* tf.contrib.layers.create_feature_spec_for_parsing(feature_columns) //prepares features config from input feature_columns. Returned can be used as an arg for features
* tf.contrib.layers.crossed_column(columns,hash_bucket_size,combiner='sum',ckpt_to_load_from=None,tensor_name_in_ckpt=None,hash_key=None) // Creates crossed collum for feature crosses
* tf.contrib.layers.embedding_column(sparse_id_column,dimension,combiner='mean',initializer=None,ckpt_to_load_from=None,tensor_name_in_ckpt=None,max_norm=None,trainable=True) 
	// EmbeddingColum for sparse data in DNN
* tf.contrib.layers.scattered_embedding_column(column_name,size,dimension,hash_key,combiner='mean',initializer=None) // Embedding collum for parameter hashing. Returns: _ScatteredEmbeddingColumn
* tf.contrib.layers.input_from_feature_columns(columns_to_tensors,feature_columns,weight_collections=None,trainable=True,scope=None,cols_to_outs=None) // Builder for FeatureColumns
* tf.contrib.layers.joint_weighted_sum_from_feature_columns(columns_to_tensors,feature_columns,num_outputs,weight_collections=None,trainable=True,scope=None) // Restricted linear prediction for featurecolumns. Computes prediction of linear model storing all weights into a single Var.
* tf.contrib.layers.make_place_holder_tensors_for_base_features(feature_columns) //Returns placeholder tensors for inference.
* tf.contrib.layers.multi_class_target(n_classes,label_name=None,weight_column_name=None) // THIS FUNCTION IS DEPRECATED
* tf.contrib.layers.one_hot_column(sparse_id_column) //_OneHotColumn for a one-hot or multi-hot repr in a DNN.
* tf.contrib.layers.parse_feature_columns_from_examples(serialized,feature_columns,name=None,example_names=None) // 
* tf.contrib.layers.parse_feature_columns_from_sequence_examples(serialized,context_feature_columns,sequence_feature_columns,name=None,example_name=None) // Parses SequenceExamples to extract tensors for featurcolums
* tf.contrib.layers.real_valued_column(column_name,dimension=1,default_value=None,dtype=tf.float32,normalizer=None) // Creates a _RealValuedColumn for dense numeric data.
* tf.contrib.layers.shared_embedding_columns(sparse_id_columns,dimension,combiner='mean',shared_embedding_name=None,initializer=None,ckpt_to_load_from=None,tensor_name_in_ckpt=None,max_norm=None,trainable=True) //
	//Creates a list of _EmbeddingColumn sharing the same embedding.
* tf.contrib.layers.sparse_column_with_hash_bucket(column_name,hash_bucket_size,combiner='sum',dtype=tf.string,hash_keys=None) 
	// Creates a _SparseColumn for hashed bucket configuration.  Use this when your sparse features are in string or integer format, but you don't have a vocab file that maps each value to an integer ID
* tf.contrib.layers.sparse_column_with_integerized_feature(column_name,bucket_size,combiner='sum',dtype=tf.int64) // Creates an integerized _SparseColumn. For use when features are pre-integerized into int64
* tf.contrib.layers.sparse_column_with_keys(column_name,keys,default_value=-1,combiner='sum',dtype=tf.string) // Creates a _SparseColumn with keys.
* tf.contrib.layers.sparse_column_with_vocabulary_file (column_name,vocabulary_file,num_oov_buckets=0,vocab_size=None,default_value=-1,combiner='sum',dtype=tf.string) // Use this when your sparse features are in string or integer format, and you have a vocab file that maps each value to an integer ID. 
* tf.contrib.layers.weighted_sparse_column(sparse_id_column,weight_column_name,dtype=tf.float32) 
	// This configuration assumes that input dictionary of model contains the following two items: * (key="sparse_col", value=sparse_tensor) where sparse_tensor is a SparseTensor. * (key="weights_col", value=weights_tensor) where weights_tensor is a SparseTensor. Following are assumed to be true: * sparse_tensor.indices = weights_tensor.indices * sparse_tensor.dense_shape = weights_tensor.dense_shape
* tf.contrib.layers.weighted_sum_from_feature_columns(columns_to_tensors,feature_columns,num_outputs,weight_collections=None,trainable=True,scope=None) // Linear prediction builder based on columsn. Function generates a weighted sum ofor each num_outputs.
* tf.contrib.layers.infer_real_valued_columns(features) //
* tf.contrib.layers.sequence_input_from_feature_columns(*args,**kwargs) //THIS FUNCTION IS EXPERIMENTAL. Builds inputs for sequence models from FeatureColumns. Returns A Tensor which can be consumed by hidden layers in the neural network.