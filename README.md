# Hybrid evolutionary and gradient descent training of deep neural networks
A novel method of training deep neural networks using both 
evolutionary and traditional approach based on gradient descent.
The idea of this repository is to create a hybrid system for 
training deep learning networks that not only traines
weights of the model but also its architecture (thanks to evolutionary part). 

After evolutionary operators are finished with their transformations
of chromosomes in a generation, the gradient descent algorithm will make 
one epoch training on the chromosomes weights.

Chromosomes store weights of each layer of the network
with previously defined settings.
Architecture is shaped by using 0 weights values in some 
share of networks connections (0.5 by default). 
Such zeros are not changed during mutation or gradient descent step.
