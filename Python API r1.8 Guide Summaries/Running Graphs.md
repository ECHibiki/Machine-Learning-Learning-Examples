Graph Library covers classes for launching graphs and operating executions. 
Covers session management. Ignores error classes. Divides into subsections

* tf.Session		//A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated. 
		# Build a graph.
		a = tf.constant(5.0)
		b = tf.constant(6.0)
		c = a * b

		# Launch the graph in a session.
		sess = tf.Session()

		# Evaluate the tensor `c`.
		print(sess.run(c))
		
		* ConfigProto protocol buffer exposes various configuration options for a session. 
				# Launch the graph in a session that allows soft device placement and
				# logs the placement decisions.
				sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
														log_device_placement=True))
														
		1) Properties:
			* graph: Current session Graph
			* graph_def: serializable version of TF graph
			* sess_str: The TF process with which the session connects
			Returns:  A graph_pb2.GraphDef proto containing nodes for all of the Operations in the underlying TensorFlow graph.
		2) Methods:
			* __init__( target='', graph=None, config=None) : Creates new TF session. 
			* __enter__() ,  __exit__(exec_type,  exec_value, exec_t)
			* as_default() : Makes certain object default setting . Must be closed explicitly
					c = tf.constant(...)
					sess = tf.Session()
					with sess.as_default():
					  print(c.eval())
					# ...
					with sess.as_default():
					  print(c.eval())

					sess.close()			
			Returns: A context manager using this session as the default session.
			* close() :  Closes session
			* list_devices() : Lists devices which can be used by session
			* make_callable( fetches, feed_list=None, accept_options=False) : Takes in values to fetch (@see session.run). Returns a function that executes that value.
			* partial_run( handle, fetches, feed_dict=None) : Experimental that partially executes that setup by partial_run_setup() activated by a partial_run()
			* partial_run_setup( fetches, feeds=None) :  Sets up a graph to be partially run
			* @staticmethod reset( target, containers=None, config=None) : Resets rescources on target and closes connected sessions.
			* run(  fetches,  feed_dict=None,  options=None,  run_metadata=None) : Runs and evalutates tensors from fetches. Runs one step of the computation evalating ever tensor in fetches and  values inserted from feed_dict.
					   a = tf.constant([10, 20])
					   b = tf.constant([1.0, 2.0])
					   # 'fetches' can be a singleton
					   v = session.run(a)
					   # v is the numpy array [10, 20]
					   # 'fetches' can be a list.
					   v = session.run([a, b])
					   # v is a Python list with 2 numpy arrays: the 1-D array [10, 20] and the
					   # 1-D array [1.0, 2.0]
					   # 'fetches' can be arbitrary lists, tuples, namedtuple, dicts:
					   MyData = collections.namedtuple('MyData', ['a', 'b'])
					   v = session.run({'k1': MyData(a, b), 'k2': [b, a]})
					   # v is a dict with
					   # v['k1'] is a MyData namedtuple with 'a' (the numpy array [10, 20]) and
					   # 'b' (the numpy array [1.0, 2.0])
					   # v['k2'] is a list with the numpy array [1.0, 2.0] and the numpy array
					   # [10, 20].
			returns: A value or list of fetches	
* tf.InteractiveSession				//for use in shells. Convinient for ipython notebooks
				sess = tf.InteractiveSession()
				a = tf.constant(5.0)
				b = tf.constant(6.0)
				c = a * b
				# We can just use 'c.eval()' without passing 'sess'
				print(c.eval())
				sess.close()
		* Properties mimic Session
		* Methods similar. See documentation for details.
* tf.get_default_session //Returns the default session for the current program thread.  Is the innermost session on which as Session or Session.as_default() has been entered

