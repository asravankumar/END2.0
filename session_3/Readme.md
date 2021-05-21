## Assignment 3 - Pytorch

### Problem Statement
Write a neural network that can take 2 inputs:
  - an image from MNIST dataset, and
  - a random number between 0 and 9
  and gives two outputs:
  - the "number" that was represented by the MNIST image, and
  - the "sum" of this number with the random number that was generated and sent as the input to the network 


![alt text](https://github.com/asravankumar/END2.0/blob/master/session_3/assign.png)


### Data Download for training
MNIST training data is downloaded from torchvision datasets.
The training data consists of 60000 labelled images.

### Data Generation
The MNIST data is combined with the following approach to generate the complete training data for the problem.
- For each image, a random number is generated, sum is calculated with the image label. the random number is converted to one-hot vector.


### Network
The network consists of the following layers.
```
Network(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=192, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=40, bias=True)
  (out): Linear(in_features=40, out_features=10, bias=True)
  (fc3): Linear(in_features=20, out_features=24, bias=True)
  (out2): Linear(in_features=24, out_features=19, bias=True)
)
```
- Each image is (1 x 28 x 28) tensor. Hence, the image has only one channel.
- It is an input to the first convolution layer with 5x5 kernel, 6 channels and 1 stride.
- The output of first convolution layer is fed to 2nd convolution layer with 5x5 kernel, 12 channels and 1 stride.
- MaxPooling is applied to both convolution layers.
- The output of second convolution layer is converted to a single vector to feed it to a linear fully connected layers.
- The output of the fully connected layers is considered as image prediction value.
- Now, the output is converted to one-hot vector and combined with random number one-hot vector (which is generated from data loader).
- This combined vector is fed to a hidden layer and subsequently output layer.


- Hence, the image prediction occurs at the end of linear fully connected layer before combining the two vectors.
- The sum prediction is done after combining. The output later has 19 features for representing from 0 to 18.


