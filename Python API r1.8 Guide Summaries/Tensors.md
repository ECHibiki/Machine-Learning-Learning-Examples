What is a tensor?

* Vectors belong to the tensor class. 
* A tensor with Ax Ay and Az is a tensor of rank 1(1 index, or one basis vector per component)
* Scalars are tensors of rank 0 as have no index or basis vectors.
* Tensors are a part of components and basis.
* A rank 2 tensor has 9 components and 9 sets of 2 basis vectors(2 indices).
* Why two though?
	* Axx Axy Axz Ayx Ayy Ayz....
	* The forces in a solid object could point in X Y or Z and on each surface there might be a force on the force(one force with three dimensions)
* A rank three tensor has 27 components.(Axxx Axyx Axzx Ayxx...)
* What makes it powerful to combine components and basis vectors?
	* All observers agree not on basis or components, but the combination of components __and__ basis vectors.
	* Basis vectors transform amoung each other in one way and the components transform to keep the combination of components and basis the same in all observers
> https://www.youtube.com/watch?v=f5liqUk0ZTw

* Tensor is seen as a generalized matrix. It lives in a structure that interacts with other mathematical entities. If one entity is changed, then the others in tensor must follow.
* Any tensor can be put into a matrix, but not every marix is a tensor.

* In deep learning terms a Weight matrix multiples with a A1 vector to give a resulting  A2 vector. 
* If we make adjustments to A1 via a RELU B, by symetry we can reobtain A2 by symetry of tensors