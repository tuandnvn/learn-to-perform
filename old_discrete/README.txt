## This is an implementation of my deprecated discretize method

This method is not fully discretized, as it mixed with a Gaussian distribution. 
The idea is that we can generate the next action using first a fully connected layer, than a Gaussian, than we can discretized the generated action.