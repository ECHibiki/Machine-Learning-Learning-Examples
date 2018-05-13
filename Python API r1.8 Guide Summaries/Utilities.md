Utilities (contrib)
Utilities for dealing with Tensors.
* tf.contrib.util.constant_value(tensor,partial=False) // Returns the constant value of a tensor. Partially evaluates a tensor then returns it's value as an ndarray
* tf.contrib.util.make_tensor_proto(values,dtype=None,shape=None,verify_shape=False) // Returns a TensorProto. Return types vary
* tf.contrib.util.make_ndarray(tensor) // Creates a numpy ndarray from tensor
* tf.contrib.util.ops_used_by_graph_def(graph_def) // a list of strings naming the op's used in a graph
* tf.contrib.util.stripped_op_list_for_graph(graph_def) // Function frinds the stripped_op_list of metagraphdef and similar protos. Retursn an OpList of Ops used by a graph