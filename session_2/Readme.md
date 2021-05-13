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
 - The Actual Output (t1 & t2)
 - Activation Function - sigmoid
 
#### Forward Propagation
 1st layer
  - h1 =w1*i1 + w2*i2
  - h2 =w3*i1 + w4*i2
  - a_h1 = sigmoid(h1) = 1/(1 + exp(-h1))
  - a_h2 = sigmoid(h2)= 1/(1 + exp(-h2))

 2nd layer
  - o1 = w5 * a_h1 + w6 * a_h2
  - o1 = w7 * a_h1 + w8 * a_h2
  - a_o1 = sigmoid(o1) = 1/(1 + exp(-o1))
  - a_o2 = sigmoid(o2) = 1/(1 + exp(-02))

 Errors
  - E1 = (1/2) * (t1 – a_o1) ^ 2
  - E2 = (1/2) * (t2 – a_o2) ^ 2
  - E_total = E1 + E2
 
