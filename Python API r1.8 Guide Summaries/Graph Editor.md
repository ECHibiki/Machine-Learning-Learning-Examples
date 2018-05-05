Graph Editor

Graph Editor library allows for modification of an existing tf.Graph instance in-place.
The author's github username is purpledog.
Graph Editor library is an attempt to allow for other kinds of editing operations, namely, rerouting and transforming.

To edit an already running graph, follow these steps:

Build a graph.
Create a session and run the graph.
Save the graph state and terminate the session
Modify the graph with the Graph Editor.
create a new session and restore the graph state
Re-run the graph with the newly created session.

**See documentation as needed**


Module: util
tf.contrib.graph_editor.make_list_of_op
tf.contrib.graph_editor.get_tensors
tf.contrib.graph_editor.make_list_of_t
tf.contrib.graph_editor.get_generating_ops
tf.contrib.graph_editor.get_consuming_ops
tf.contrib.graph_editor.ControlOutputs
tf.contrib.graph_editor.placeholder_name
tf.contrib.graph_editor.make_placeholder_from_tensor
tf.contrib.graph_editor.make_placeholder_from_dtype_and_shape
Module: select
tf.contrib.graph_editor.filter_ts
tf.contrib.graph_editor.filter_ts_from_regex
tf.contrib.graph_editor.filter_ops
tf.contrib.graph_editor.filter_ops_from_regex
tf.contrib.graph_editor.get_name_scope_ops
tf.contrib.graph_editor.check_cios
tf.contrib.graph_editor.get_ops_ios
tf.contrib.graph_editor.compute_boundary_ts
tf.contrib.graph_editor.get_within_boundary_ops
tf.contrib.graph_editor.get_forward_walk_ops
tf.contrib.graph_editor.get_backward_walk_ops
tf.contrib.graph_editor.get_walks_intersection_ops
tf.contrib.graph_editor.get_walks_union_ops
tf.contrib.graph_editor.select_ops
tf.contrib.graph_editor.select_ts
tf.contrib.graph_editor.select_ops_and_ts
Module: subgraph
tf.contrib.graph_editor.SubGraphView
tf.contrib.graph_editor.make_view
tf.contrib.graph_editor.make_view_from_scope
Module: reroute
tf.contrib.graph_editor.swap_ts
tf.contrib.graph_editor.reroute_ts
tf.contrib.graph_editor.swap_inputs
tf.contrib.graph_editor.reroute_inputs
tf.contrib.graph_editor.swap_outputs
tf.contrib.graph_editor.reroute_outputs
tf.contrib.graph_editor.swap_ios
tf.contrib.graph_editor.reroute_ios
tf.contrib.graph_editor.remove_control_inputs
tf.contrib.graph_editor.add_control_inputs
Module: edit
tf.contrib.graph_editor.detach_control_inputs
tf.contrib.graph_editor.detach_control_outputs
tf.contrib.graph_editor.detach_inputs
tf.contrib.graph_editor.detach_outputs
tf.contrib.graph_editor.detach
tf.contrib.graph_editor.connect
tf.contrib.graph_editor.bypass
Module: transform
tf.contrib.graph_editor.replace_t_with_placeholder_handler
tf.contrib.graph_editor.keep_t_if_possible_handler
tf.contrib.graph_editor.assign_renamed_collections_handler
tf.contrib.graph_editor.transform_op_if_inside_handler
tf.contrib.graph_editor.copy_op_handler
tf.contrib.graph_editor.Transformer
tf.contrib.graph_editor.copy
tf.contrib.graph_editor.copy_with_input_replacements
tf.contrib.graph_editor.graph_replace
