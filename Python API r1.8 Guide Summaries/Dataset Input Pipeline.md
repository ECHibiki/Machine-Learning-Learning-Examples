Dataset Input Pipeline 
Reader classes
Classes that create a dataset from input files.
 * tf.data.Dataset // A Dataset can be used to represent an input pipeline as a collection of elements (nested structures of tensors) and a "logical plan" of transformations that act on those elements.
 * tf.data.FixedLengthRecordDataset //  CLASS A Dataset comprising lines from one or more text files.
 * tf.data.TextLineDataset // CLASS A Dataset comprising records from one or more  * tf.ecord files.
 * tf.data.TFRecordDataset //  CLASS A Dataset comprising records from one or more  * tf.ecord files.

Creating new datasets
Static methods in Dataset that create new datasets.

 * tf.data.Dataset.from_generator(generator,output_types,output_shapes=None)  // Dataset whose elements are created by a generator
				 def gen():
				  for i in itertools.count(1):
					yield (i, [1] * i)

				ds = Dataset.from_generator(
					gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
				value = ds.make_one_shot_iterator().get_next()

				sess.run(value)  # (1, array([1]))
				sess.run(value)  # (2, array([1, 1]))
 * tf.data.Dataset.from_tensor_slices(tensors) //Dataset whose elements are slices of the given tensors.
 * tf.data.Dataset.from_tensors(tensors) //Dataset with a single element, comprising the given tensors.
 * tf.data.Dataset.list_files(file_pattern,shuffle=None) // dataset of all files matching a pattern. RTN A Dataset of strings corresponding to file names.
 * tf.data.Dataset.range(*args) // Creates a Dataset of a step-separated range of values. RTN Dataset: A RangeDataset.
 * tf.data.Dataset.zip(datasets) // Dataset by zipping together the given datasets.
 
			 # NOTE: The following examples use `{ ... }` to represent the
			# contents of a dataset.
			a = { 1, 2, 3 }
			b = { 4, 5, 6 }
			c = { (7, 8), (9, 10), (11, 12) }
			d = { 13, 14 }

			# The nested structure of the `datasets` argument determines the
			# structure of elements in the resulting dataset.
			Dataset.zip((a, b)) == { (1, 4), (2, 5), (3, 6) }
			Dataset.zip((b, a)) == { (4, 1), (5, 2), (6, 3) }

			# The `datasets` argument may contain an arbitrary number of
			# datasets.
			Dataset.zip((a, b, c)) == { (1, 4, (7, 8)),
										(2, 5, (9, 10)),
										(3, 6, (11, 12)) }

			# The number of elements in the resulting dataset is the same as
			# the size of the smallest dataset in `datasets`.
			Dataset.zip((a, d)) == { (1, 13), (2, 14) }


Transformations on existing datasets
These functions transform an existing dataset, and return a new dataset. Calls can be chained together, as shown in the example below:

train_data = train_data.batch(100).shuffle().repeat()
 * tf.data.Dataset.apply(transformation_func) // applies a transformation to dataset
 * tf.data.Dataset.batch(batch_size)  //Combines consecutive elements of this dataset into batches.
 * tf.data.Dataset.cache(filename='') //Caches the elements in this dataset.
 * tf.data.Dataset.concatenate(dataset) // combines two into one [a.concatenate(b)]
 * tf.data.Dataset.filter(predicate) // Filters this dataset according to predicate. predicate: a fn mapping a nested structure of tensors  to a scalar tf.bool tensor.
 * tf.data.Dataset.flat_map(map_func)//Maps map_func across this dataset and flattens the result.
 * tf.data.Dataset.interleave(map_func,cycle_length,block_length=1) //
				 # NOTE: The following examples use `{ ... }` to represent the
				# contents of a dataset.
				a = { 1, 2, 3, 4, 5 }

				# NOTE: New lines indicate "block" boundaries.
				a.interleave(lambda x: Dataset.from_tensors(x).repeat(6),
							 cycle_length=2, block_length=4) == {
					1, 1, 1, 1,
					2, 2, 2, 2,
					1, 1,
					2, 2,
					3, 3, 3, 3,
					4, 4, 4, 4,
					3, 3,
					4, 4,
					5, 5, 5, 5,
					5, 5,
				}
* tf.data.Dataset.map(map_func,num_parallel_calls=None) // Maps function accross dataset
* tf.data.Dataset.padded_batch(batch_size,padded_shapes,padding_values=None) // Combines elements of dataset into padded batches
* tf.data.Dataset.prefetch(buffer_size) // Creates a Dataset that prefetches elements from this dataset.
* tf.data.Dataset.repeat(count=None) // Repeats this dataset count times.
* tf.data.Dataset.shard(num_shards,index) // Creates a Dataset that includes only 1/num_shards of this dataset. Useful when running distributed training and allows workers to read unique subsets
* tf.data.Dataset.shuffle(buffer_size,seed=None,reshuffle_each_iteration=None) // Random suffle of dataset
* tf.data.Dataset.skip(count) // skips count elements from dataset
* tf.data.Dataset.take(count) // creates dataset with at most count elements
Custom transformation functions
Custom transformation functions can be applied to a Dataset using  * tf.data.Dataset.apply. Below are custom transformation functions from  * tf.contrib.data:

* tf.contrib.data.batch_and_drop_remainder(batch_size) // A batching transformation that omits the final small batch (if present).
* tf.contrib.data.dense_to_sparse_batch(batch_size,row_shape) //  Transformation that batches ragged elements to sparsetensors
* tf.contrib.data.enumerate_dataset(start=0) // A transformation that enumerate the elements of a dataset.
* tf.contrib.data.group_by_window(key_func,reduce_func,window_size=None,window_size_func=None) // Transform to group windows of elements by key
* tf.contrib.data.ignore_errors() // dataset created with ignored errors
* tf.contrib.data.map_and_batch(map_func,batch_size,num_parallel_batches=1,drop_remainder=False) // Fused implementation of map and batch.
* tf.contrib.data.padded_batch_and_drop_remainder(batch_size,padded_shapes,padding_values=None) // Batching and padding transform to omit the final small batch
* tf.contrib.data.parallel_interleave(map_func,cycle_length,block_length=1,sloppy=False,buffer_output_elements=None,prefetch_input_elements=None) 
	//parallel version of the Dataset.interleave() transformation.
* tf.contrib.data.rejection_resample(class_func,target_dist,initial_dist=None,seed=None) // Tranform that resamples datset to achieve target distrubution
* tf.contrib.data.scan(initial_state,scan_func) // scans accros dataset returning a transformation function to pass onto apply
* tf.contrib.data.shuffle_and_repeat(buffer_size,count=None,seed=None) // Shuffles and repeats a Dataset returning a new permutation for each epoch.
* tf.contrib.data.unbatch() // splits the elements of a dataset. if elements of the dataset are shaped [B, a0, a1, ...], where B may vary from element to element, then for each element in the dataset, the unbatched dataset will contain B consecutive elements of shape [a0, a1, ...].

Iterating over datasets
These functions make a  * tf.data.Iterator from a Dataset.
* tf.data.Dataset.make_initializable_iterator(shared_name=None) // creates an itterator to enumerate accross the dataset
* tf.data.Dataset.make_one_shot_iterator() // The returned iterator will be initialized automatically. A "one-shot" iterator does not currently support re-initialization.

The Iterator class also contains static methods that create a  * tf.data.Iterator that can be used with multiple Dataset objects.
* tf.data.Iterator.from_structure(output_types,output_shapes=None,shared_name=None,output_classes=None) //New unititialized itterator with given structer. Can create a new reusable itterator
* tf.data.Iterator.from_string_handle(string_handle,output_types,output_shapes=None,output_classes=None) // new, uninitialized Iterator based on the given handle. Allows you to define a "feedable" iterator where you can choose between concrete iterators by feeding a value in a tf.Session.run call.
Extra functions from  * tf.contrib.data
* tf.contrib.data.get_single_element (dataset) // Returns the single element in dataset as a nested structure of tensors.
* tf.contrib.data.make_saveable_from_iterator(iterator) // Returns a SaveableObject for saving/restore iterator state using Saver.
* tf.contrib.data.read_batch_features(file_pattern,batch_size,features,reader=tf.data.TFRecordDataset,reader_args=None,randomize_input=True,num_epochs=None,capacity=10000) 
	// THIS FUNCTION IS DEPRECATED tf.contrib.data.make_batched_features_dataset

