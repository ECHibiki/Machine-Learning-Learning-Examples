Higher Order Operators
TensorFlow provides several higher order operators to simplify the common map-reduce programming patterns.
* tf.map_fn(fn,elems,dtype=None,parallel_iterations=10,back_prop=True,swap_memory=False,infer_shape=True,name=None)
	// Does map function on each element.
		elems = np.array([1, 2, 3, 4, 5, 6])
		squares = map_fn(lambda x: x * x, elems)
		# squares == [1, 4, 9, 16, 25, 36]
* tf.foldl(fn,elems,initializer=None,parallel_iterations=10,back_prop=True,swap_memory=False,name=None) 
	// This foldl operator repeatedly applies the callable fn to a sequence of elements from first to last. 
		elems = [1, 2, 3, 4, 5, 6]
		sum = foldl(lambda a, x: a + x, elems)
		# sum == 21
* tf.foldr(fn,elems,initializer=None,parallel_iterations=10,back_prop=True,swap_memory=False,name=None)
	// Same as foldl, but last to first
* tf.scan(fn,elems,initializer=None,parallel_iterations=10,back_prop=True,swap_memory=False,infer_shape=True,name=None)
	// Calls on every item using the accumalation on every result from first to last. Returns a sequence of tensors.
		elems = np.array([1, 0, 0, 0, 0, 0])
		initializer = (np.array(0), np.array(1))
		fibonaccis = scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
		# fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])