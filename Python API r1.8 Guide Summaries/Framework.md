Framework

Framework utilities.
* tf.convert_to_tensor_or_sparse_tensor(value,dtype=None,name=None) // Converts to a SpareTensor or Tensor base on the input value
* tf.contrib.framework.get_graph_from_inputs(op_input_list,graph=None) // Returns graph for given inputs
* tf.is_numeric_tensor(tensor) 
* tf.is_non_decreasing( x,  name=None) // True tensor if X non decreasing
* tf.is_strictly_increasing(x,name=None) //  strictly increasing if for every adjacent pair we have x[i] < x[i+1]
* tf.contrib.framework.is_tensor // Checktype for tensor 
* tf.contrib.framework.reduce_sum_n(tensors,name=None) // Reduces all tensors by a sum via  via tf.reduce_sum then tf.add_n.
* tf.contrib.framework.remove_squeezable_dimensions (predictions,labels,name=None) // THIS FUNCTION IS DEPRECATED
* tf.contrib.framework.with_shape(expected_shape,tensor) 		  //If tensor shape and expected_shape, are fully defined, assert they match. Otherwise, add assert op that will validate the shape when tensor is evaluated, and set shape on tensor.
* tf.contrib.framework.with_same_shape(expected_tensor,  tensor)		//Assert tensors are the same shape, from the same graph.
Variables
* tf.contrib.framework.add_model_variable(var) // Adds a variable to the GraphKeys.MODEL_VARIABLES collection.
* tf.train.assert_global_step(global_step_tensor)
* tf.contrib.framework.assert_or_get_global_step(graph=None, global_step_tensor=None)
* tf.contrib.framework.assign_from_checkpoint(model_path,var_list,ignore_missing_vars=False)//Creates an operation to assign specific variables from a checkpoint.
* tf.contrib.framework.assign_from_checkpoint_fn(model_path,var_list,ignore_missing_vars=False,reshape_variables=False) //Returns a function that assigns specific variables from a checkpoint. function that takes a single argument, a tf.Session, that applies the assignment operation.
* tf.contrib.framework.assign_from_values(var_names_to_values) // Creates an assignment operation from a given mapping. unction provides a mechanism for performing assignment of variables to values in a way that does not fill the graph with large assignment values.
* tf.contrib.framework.assign_from_values_fn(var_names_to_values) //Returns a function that assigns specific variables from the given values. This function provides a mechanism for performing assignment of variables to values in a way that does not fill the graph with large assignment values.
* tf.contrib.framework.create_global_step(graph=None) // THIS FUNCTION IS DEPRECATED.
* tf.contrib.framework.filter_variables(var_list,include_patterns=None,exclude_patterns=None,reg_search=True) // Filter a list of variables using regular expressions.
* tf.train.get_global_step(graph=None) //The global step variable, or None if none was found.
* tf.contrib.framework.get_or_create_global_step(graph=None) //THIS FUNCTION IS DEPRECATED. 
* tf.contrib.framework.get_local_variables(scope=None,suffix=None) //Gets the list of local variables, filtered by scope and/or suffix.
* tf.contrib.framework.get_model_variables(scope=None,suffix=None) //Gets the list of model variables, filtered by scope and/or suffix.
* tf.contrib.framework.get_unique_variable(var_op_name)	// take the full name of the variable op, including the scope. Gets the variable uniquely identified by that var_op_name.
* tf.contrib.framework.get_variables_by_name(given_name,scope=None) //Gets the list of variables that were given that name.
* tf.contrib.framework.get_variables_by_suffix(suffix,scope=None) // Gets the list of variables that end with the given suffix.
* tf.contrib.framework.get_variables_to_restore(include=None,exclude=None) //Gets the list of the variables to restore.
* tf.contrib.framework.get_variables(scope=None,suffix=None,collection=tf.GraphKeys.GLOBAL_VARIABLES) //Gets the list of variables, filtered by scope and/or suffix.
* tf.contrib.framework.local_variable(initial_value,validate_shape=True,name=None,use_resource=None)  //Create a variable with a value and add it to GraphKeys.LOCAL_VARIABLES. Returns a variable.
* tf.contrib.framework.model_variable(name,shape=None,dtype=tf.float32,initializer=None,regularizer=None,trainable=True,collections=None,caching_device=None,device=None,partitioner=None,custom_getter=None,use_resource=None)
	//Gets an existing model variable with these parameters or creates a new one.
* tf.contrib.framework.variable(name,shape=None,dtype=None,initializer=None,regularizer=None,trainable=True,collections=None,caching_device=None,device=None,partitioner=None,custom_getter=None,use_resource=None)
	//Gets or creates a new variable with parameters
* tf.contrib.framework.VariableDeviceChooser // Class will assign varaibles in a round robin fasion.
* tf.contrib.framework.zero_initializer(ref,use_locking=True,name='zero_initializer') // Initializes ref with zeros. Designed to save initialization memory
Checkpoint utilities
* tf.contrib.framework.load_checkpoint(filepattern) // Returns CheckpointReader for latest checkpoint.
* tf.contrib.framework.list_variables(checkpoint_dir) // returns list of name and shape of variables at checkpoint
* tf.contrib.framework.load_variable(checkpoint_dir,name) // Get tensor with contents at checkpoint
* tf.contrib.framework.init_from_checkpoint(checkpoint_dir, assignment_map) // Initializes current variables with loaded tensors