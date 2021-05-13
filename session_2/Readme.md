## Assignment 2

### Screenshot of the Backpropagation Excel Sheet
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/1.png)
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/2.png)

### Error graphs when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
#### Learning Rate 0.1
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/Error_graph_LR_0.1.png)
#### Learning Rate 0.2
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/Error_graph_LR_0.2.png)
#### Learning Rate 0.5
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/Error_graph_LR_0.5.png)
#### Learning Rate 0.8
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/Error_graph_LR_0.8.png)
#### Learning Rate 1.0
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/Error_graph_LR_1.0.png)
#### Learning Rate 2.0
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/Error_graph_LR_2.0.png)

### Major Steps Involved
#### The Network
![alt text](https://github.com/asravankumar/END2.0/blob/master/session_2/network.png)

 - The network considered in the excel sheet takes two inputs. (i1 & i2)
 - Hidden Layer with 2 nodes
 - Output Layer with 2 nodes
 - The Actual Output (t1 & t2)
 - Activation Function on hidden layer and output layer nodes- sigmoid 
 - 8 weights w1, w2, w3, ..., w8 connecting different nodes in the network
 
#### Forward Propagation
 Based on the weights and input in the network, the input to subsequent layers are computed.

 1st layer
 ```
  - h1 =w1*i1 + w2*i2
  - h2 =w3*i1 + w4*i2
  - a_h1 = sigmoid(h1) = 1/(1 + exp(-h1))
  - a_h2 = sigmoid(h2)= 1/(1 + exp(-h2))
```

 2nd layer
 ```
  - o1 = w5 * a_h1 + w6 * a_h2
  - o1 = w7 * a_h1 + w8 * a_h2
  - a_o1 = sigmoid(o1) = 1/(1 + exp(-o1))
  - a_o2 = sigmoid(o2) = 1/(1 + exp(-02))
```

 Now that the predict value is computed, we compute the error.
 Errors w.r.t the targets. - L2 error.
 The difference between actual and predicted error.
 ```
  - E1 = (1/2) * (t1 – a_o1) ^ 2
  - E2 = (1/2) * (t2 – a_o2) ^ 2
  - E_total = E1 + E2
 ```

#### BackPropagation
According to Gradient Descent optimization algorithm, the weights are updated based on the below equation

```
 - wi = wi - learning_rate * d(E_t)/dwi
 - dE_t/dwi is a partial derivate of Total error w.r.t the weight wi.
```

Hence, multiple iterations have to be performed on the network for the weights to be updated to reach minimal error.

Backpropagation calculates the gradients of errors with respect to different weights/parameters of the network used to obtain optimal weights of the network.

Let's start with updating w5. We need to compute dE_t/dw5.
Based on the chain rule, if x effects y, y effects z. Then d(z)/d(x) = d(z)/d(y) * d(y)/d(x)

```
dE_t/dw5 = d(E1 + E2)/dw5
dE_t/dw5 = dE1/dw5 (as E2 is not dependent on w5)
dE_t/dw5 = dE1/da_o1 * da_o1/do1 * do1/dw5 (by applying calculus chain rule - E1 dependent on a_o1, a_o1 dependent on o1)
	- dE_1/da_o1 = d(1/2 * (t1 - a_o1)^2)/da_o1 = (t1 - a_o1) * (-1) = (a_o1 - t1)
	- da_o1/do1 = d(sigmoid(o1))/do1 = sigmoid(o1) * (1 - sigmoid(o1)) = a_o1 * (1 - a_o1)
	- do1/dw5 = d(w5 * a_h1 + w6 * a_h2)/dw5 = a_h1
Hence,
dE_t/dw5 = (a_o1 - t1) * (a_o1) * (1 - a_o1) * a_h1
On similar lines,
dE_t/dw6 = (a_o1 - t1) * (a_o1) * (1 - a_o1) * a_h2
dE_t/dw7 = (a_o2 - t2) * (a_o2) * (1 - a_o2) * a_h1
dE_t/dw8 = (a_o2 - t2) * (a_o2) * (1 - a_o2) * a_h2
```

```
dE_t/da_h1 = d(E1+E2)/da_h1 = dE1/da_h1 + dE2/da_h1
```
By applying chain rule. as a_h1 affects E1 in the following route. a_h1 -> o1 -> a_o1 -> E1 
and from E1, da_h1 can only be reached from a_o1. 
```
dE1/da_h1 = dE1/da_o1 * da_o1/do1 * do1/da_h1 = (a_o1 - t1) * (a_o1) * (1 - a_o1) * w5
dE2/da_h1 = dE2/da_o2 * da_o2/do2 * do2/da_h1 = (a_o2 - t2) * (a_o2) * (1 - a_o2) * w7

dE1/da_h2 = dE1/da_o1 * da_o1/do1 * do1/da_h2 = (a_o1 - t1) * (a_o1) * (1 - a_o1) * w6
dE2/da_h2 = dE2/da_o2 * da_o2/do2 * do2/da_h1 = (a_o2 - t2) * (a_o2) * (1 - a_o2) * w8
```

Let's compute the gradient's for w1.
Based on the network, we can see that w1 only affects a_h1.
E_Total = (E1 + E2)
Hence,
dE_t/dw1 = dE1/dw1 + dE2/dw1
E1 is directly affected by w1 in the following route. w1 -> h1 -> a_h1 -> o1 -> a_o1 -> E1
E2 is affected by w1 in the following route. w1 -> h1 -> a_h1 -> o2 -> a_o2 -> E2
Hence we get the following by applying chain rule.

```
dE_t/dw1 = dE1/da_o1 * da_o1/do1 * do1/da_h1 * da_h1/dh1 * dh1/dw1 + dE2/da_o2 * da_o2/do2 * do2/da_h1 * da_h1/dh1 * dh1/dw1
         = (dE1/da_h1 + dE2/da_h1) * da_h1/dh1 * dh1/dw1
         = (dE_t/da_h1) * da_h1/dh1 * dh1/dw1 
         = (dE_t/da_h1) * (a_h1) * (1 - a_h1) * i1
```

By replacing the dE_t/da_h1, we get the complete equation
```
dE_t/dw1 = ((a_o1 - t1) * (a_o1) * (1 - a_o1) * w5 + (a_o2 - t2) * (a_o2) * (1 - a_o2) * w7) * (a_h1) * (1 - a_h1) * i1
```

Similarly, the same can be computed for w2, w3, w4
```
dEt/dw2  = (dE_t/da_h1) * (a_h1) * (1 - a_h1) * i2
dEt/dw3  = (dE_t/da_h2) * (a_h2) * (1 - a_h2) * i1
dEt/dw4  = (dE_t/da_h2) * (a_h2) * (1 - a_h2) * i2
```
And now the complete equations
```
dE_t/dw2 = ((a_o1 - t1) * (a_o1) * (1 - a_o1) * w5 + (a_o2 - t2) * (a_o2) * (1 - a_o2) * w7) * (a_h1) * (1 - a_h1) * i2
dE_t/dw3  = ((a_o1 - t1) * (a_o1) * (1 - a_o1) * w6 + (a_o2 - t2) * (a_o2) * (1 - a_o2) * w8) * (a_h2) * (1 - a_h2) * i1
dE_t/dw4  = ((a_o1 - t1) * (a_o1) * (1 - a_o1) * w6 + (a_o2 - t2) * (a_o2) * (1 - a_o2) * w8) * (a_h2) * (1 - a_h2) * i2
```

All these weights are updated based on the gradient descent formula
```
w_new = w_old - learning_rate * dE_total/dw_old
```
The updating when done with appropriate learning rate would yield weights with minimal error.
