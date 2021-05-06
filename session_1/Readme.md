## Assignment
# What is a neural network neuron?
  A neuron is the most basic element part of a neural network. It is a mathematical function that receives set of input values and computes a weighted sum using set of weights. This weighted sum is then passed to a non-linear activation function to form the output.This is done to remove the linearity in the model. 
  Multiple neurons organized in layers is a neural network. A neuron can be visualized as a temporary storage unit which computes based on the input values, weights and activation function and passes the output to the next layer.

# What is the use of the learning rate?
  Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect to the loss gradient while moving towards minimum of the loss function.
  The choice of the value for learning rate can impact two things: 
      1) how fast the algorithm learns 
      2) whether the cost function is minimized or not
  Lower the value, more time it takes to converge to the local minima. 
  Larger the value, it can cause the model to converge to suboptimal solution.
  So, an appropriate learning rate has to be chosen to reach optical solution with minimum loss.
# How are weights initialized?
  Weights are initialized from a normal/Gaussian distribution.
  Multiple other weights initialization techniques like Xavier initialization are used.
# What is "loss" in a neural network?
  Loss in a neural network is the difference between the true value and the predicted value from the network. Ex:- Mean Squared Error.
# What is the "chain rule" in gradient flow?
  In neural network, during backpropation, the gradient of the loss is propagated backwards in order to update the weights.
  In calculus, if x effects y, y effects z. Then d(z)/d(x) = d(z)/d(y) * d(y)/d(x). Where d() is a partial derivative.
  So, while backpropagating, since each layer is dependent on previous layer, we apply the chain rule to calculate the respective partial derivate.
