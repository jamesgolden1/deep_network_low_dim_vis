# curvature

“…Trained networks… contract space at the center of decision volumes and expand space in the vicinity of decision boundaries” – Nayebi & Ganguli, 2017

y = sig(W*x), z = sig(V*y) undergoing supervised training to separate blue and red, adapted from Olshausen, 2010. Inspired by Fig 6 of [Olshausen & Field 2005](http://www.rctn.org/bruno/CTBP/olshausen-field05.pdf) and [Olah 2014](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/). 

In order to correctly classify these data points in a 2D space as red or blue, a supervised two-layer network needs three feature vectors in the first layer (W; green, magenta and yellow) (Olah, 2014). If the projections onto those three feature vectors are plotted in a representation space where the features are orthogonal, we can observe how the network learns to warp the 2D input space lattice in 3D representation space in order to make classification possible with a single plane (blue cyan points), where projection onto a vector normal to the cyan plane separates the two classes (rightmost panel).

Learning feature vectors in W is equivalent to warping the input space within the representation space, and learning feature vector V is the placement of the separating plane (cyan points). Convolutional networks do something this in high dimensions to classify images, and measurements of the local curvature of the input space reveal how individual neurons and layers contribute to successful recognition.

The network is described by these equations for the two layers: y = sig(W*x), z = sig(V*y), where sig is a sigmoid nonlinearity.

![](two_layer_warping_2342403.gif)

When the classes are randomized, a deep network (with five layers of 64 units, projected back into a final layer of 3 units) is powerful enough to "memorize" a dataset by extreme warping of the input data space.

![](three_layer_warping_1511080.gif)

For data in a 3D input space, the process is similar:

![](three_layer_warping_212485.gif)



