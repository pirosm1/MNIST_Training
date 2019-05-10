# MNIST_Training

CNN Training for [MNIST](http://yann.lecun.com/exdb/mnist/) data set recognition.

Recompile with the "verbose" constant = true if you want to see timings.

If back-propagation feels slow and you've got a decent machine try to set optimization flags when compiling.

I used `/Ox` in Debug builds resulting in an average 60ms per back-propagation.

Using the default Release build resulted in about average 32ms per back-propagation.
