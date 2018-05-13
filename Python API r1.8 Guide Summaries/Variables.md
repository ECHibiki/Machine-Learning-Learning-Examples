Variables

Mostly basic stuff.

* tf.Variable // Class. Variable maintains state in graph. Requires initial value. 
__init__(
    initial_value=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None,
    constraint=None
)

Variable helper functions
TensorFlow provides a set of functions to help manage the set of variables collected in the graph.
* tf.global_variables(scope=None) // Sharec accross machine evn
* tf.local_variables // 
* tf.model_variables // 
* tf.trainable_variables // 
* tf.moving_average_variables // 
* tf.global_variables_initializer // 
* tf.local_variables_initializer // 
* tf.variables_initializer // 
* tf.is_variable_initialized // 
* tf.report_uninitialized_variables // 
* tf.assert_variables_initialized // 
* tf.assign(
    ref,
    value,
    validate_shape=None,
    use_locking=None,
    name=None
) // Updates a ref by assigning it a vlue
* tf.assign_add(
    ref,
    value,
    use_locking=None,
    name=None
) // Same but adding
* tf.assign_sub(
    ref,
    value,
    use_locking=None,
    name=None
) // same but subtracking

Saving and Restoring Variables
* tf.train.Saver // 
* tf.train.latest_checkpoint // 
* tf.train.get_checkpoint_state // 
* tf.train.update_checkpoint_state //
