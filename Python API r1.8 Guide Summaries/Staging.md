Staging:

* tf.contrib.staging.StagingArea 
	// Class StagingArea is datastructure storing tensors from multiple steps exposing operations that can get and put tensors. 
	// Staging Areas is a tupple of tensors and capacity variable or static.
	* Properties: 
		* capacity : maximum number of elements of this staging area.
		* dtypes :  The list of dtypes for each component of a staging area element.
		* memory_limit : 	The maximum number of bytes of this staging area. 0 implies unbound
		* name : 	The name of the staging area.
		* names : 	The list of names for each component of a staging area element.
		* shapes : 	The list of shapes for each component of a staging area element.
	* Methods:
		* __init__(dtypes,shapes=None,names=None,shared_name=None,capacity=0,memory_limit=0) :  Creates
		* clear(name=None) : clears the StagingArea
		* get(name=None) : gets a tupple of tensors. Operation can be named.
		* peek(index,name=None) : Peeks at an element in the staging area
		* put(values,name=None) : Creates an opperation to place a value in the staging area
		* size(name=None) ; Returns number of elements in the staging area