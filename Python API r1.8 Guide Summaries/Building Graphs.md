Building Graphs 

TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. This leads to a low-level programming model in which you first define the dataflow graph, then create a TensorFlow session to run parts of the graph across a set of local and remote devices.

This guide will be most useful if you intend to use the low-level programming model directly. Higher-level APIs such as tf.estimator.Estimator and Keras hide the details of graphs and sessions from the end user, but this guide may also be useful if you want to understand how these APIs are implemented.

** Read Docs as needed ** 

Core graph data structures 
 * tf.Graph // 
 * tf.Operation // 
 * tf.Tensor // 
Tensor types
 * tf.DType // 
 * tf.as_dtype // 
Utility functions
 * tf.device // 
 * tf.container // 
 * tf.name_scope // 
 * tf.control_dependencies // 
 * tf.convert_to_tensor // 
 * tf.convert_to_tensor_or_indexed_slices // 
 * tf.convert_to_tensor_or_sparse_tensor // 
 * tf.get_default_graph // 
 * tf.reset_default_graph // 
 * tf.import_graph_def // 
 * tf.load_file_system_library // 
 * tf.load_op_library // 
Graph collections
 * tf.add_to_collection // 
 * tf.get_collection // 
 * tf.get_collection_ref // 
 * tf.GraphKeys // 
Defining new operations
 * tf.RegisterGradient // 
 * tf.NotDifferentiable // 
 * tf.NoGradient // 
 * tf.TensorShape // 
 * tf.Dimension // 
 * tf.op_scope // 
 * tf.get_seed // 
For libraries building on TensorFlow
 * tf.register_tensor_conversion_function //