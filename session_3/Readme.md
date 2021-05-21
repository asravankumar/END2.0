## Assignment 3 - Pytorch

### Problem Statement
Write a neural network that can take 2 inputs:
  - an image from MNIST dataset, and
  - a random number between 0 and 9
  and gives two outputs:
  - the "number" that was represented by the MNIST image, and
  - the "sum" of this number with the random number that was generated and sent as the input to the network 


![alt text](https://github.com/asravankumar/END2.0/blob/master/session_3/assign.png)


### check device
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```

### Data Download for training
MNIST training data is downloaded from torchvision datasets.
The training data consists of 60000 labelled images.

```
# download training mnist dataset which consists of 60000 labelled images
train_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor() # converts the images into pytorch tensors
    ])
)
```

### Data Generation
The MNIST data is combined with the following approach to generate the complete training data for the problem.
- For each image, a random number is generated, sum is calculated with the image label. 
- The random number is converted to one-hot vector.
- Hence, there are two inputs - image & one-hot vector of random number.
- There are two outputs - the image label & other being the sum.

```
# data set creation
class MNISTAdderDataset(Dataset):
  # Dataset to create mnist custom data for mnist + random number adder.

  def __init__(self, dataset):
    self.mnist_data = dataset
    
  def __getitem__(self, index):
    # returns tuple with following input and output for every image.
    # input : x1 - tensor image
    # input : x2 - one hot vector for random number
    # labelled output : y1 - the image's label number.
    # labelled output : y2 - the sum.

    x1, y1 = self.mnist_data[index]
    x2 = random.randint(0,9)
    y2 = y1 + x2 
    return (x1, F.one_hot(torch.tensor(x2), 10).float(), y1, y2)

  def __len__(self):
    return len(self.mnist_data)
```

```
mnist_adder_training_data = MNISTAdderDataset(train_set)
mnist_adder_training_data_loader = torch.utils.data.DataLoader(mnist_adder_training_data,
                                               batch_size=100,
                                               shuffle=False,
                                               )
```

### Network
The network consists of two parts.
- Image classification.
  - This is done by making use of two convolution layers and two linear fully connected layers.
- Sum computing.
  - The input is a combined vector of the image output vector to one-hot vector of random number.
  - Fully connected layers with 19 output features.(one for each sum value 0 to 18)

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

```
import torch.nn as nn
import torch.nn.functional as F

# Network creation

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # input is 28x28 with 1 channel, kernel size is 5x5
        # output of convolution layer is 24x24 i.e (28 - 5 )/ 1 + 1
        # maxpool layer is with kernel size 2 and stride 2
        # output of max pool layer is 12x12 i.e (24 - 2)/2 + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # input is 12x12 with 1 channel, kernel size is 5x5
        # output of convolution layer is 8x8 i.e (12 - 5 )/ 1 + 1
        # maxpool layer is with kernel size 2 and stride 2
        # output of max pool layer is 4x4 i.e (8 - 2)/2 + 1
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # fully connected layer with input will be 12 * 4 * 4
        # and 50 output features
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=50)
        # another fully connected layer
        self.fc2 = nn.Linear(in_features=50, out_features=40)
        # output layer for image classification
        self.out = nn.Linear(in_features=40, out_features=10)

        # fully connected layer after combining image classification output and random number one-hot vector.
        self.fc3 = nn.Linear(in_features=20, out_features=21)
        #self.fc4 = nn.Linear(in_features=30, out_features=24)

        # final sum output layer. 19 features output. 19 because of one-hot vector for 0 to 19. as sum of 0 - 9 image  with 0 - 9 random numbers.
         self.out2 = nn.Linear(in_features=21, out_features=19)

    def forward(self, t, rand_num_vector):
        # convolution layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # convolution layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # hidden linear layer
        # converting the image to a straight vector.
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # output layer for image classification
        t = self.out(t)
        
        # converting to one hot vector for image predicted value.
        t_vector = F.one_hot(t.argmax(dim=1), num_classes=10)
        #print("t_vector", t_vector)
        #print("rand_num_vector", rand_num_vector)

        # now combining the image output with random number vector for addition.
        combined_vector = torch.cat([t_vector, rand_num_vector], dim=-1)
        #print("combined vector", combined_vector)
        combined_vector = self.fc3(combined_vector)
        combined_vector = F.relu(combined_vector)

        #combined_vector = self.fc4(combined_vector)
        #combined_vector = F.relu(combined_vector)


        # final layer with outputs the sum of the two numbers.
        combined_vector = self.out2(combined_vector)
        combined_vector = F.relu(combined_vector)

        return t, combined_vector
```

We can see above how the random number vector and the prediction vector after image classification is done.

```
 combined_vector = torch.cat([t_vector, rand_num_vector], dim=-1)
```


### Training
Training is done by using the data loader with batch size of 100.
Loss is calculated for image classification and sum prediction individually and is propagated back. 
Adam Optimizer is used during training.

```
import torch.optim as optim
torch.set_grad_enabled(True) # to enable the gradients.

network = Network().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.02) # using adam's optimizer with learning rate of 0.02

# to check the number of correct predictions.
def get_num_correct(X, Y):
  return X.argmax(dim=1).eq(Y).sum().item()



for epoch in range(10):

    total_loss = 0
    total_correct_mnist = 0
    total_correct_sum = 0

    for batch in mnist_adder_training_data_loader:
        # for every batch predict using network and backpropagate the loss.
        # x1s : images input
        # x2s : random number one hot vectors.
        # y1s : image labels.
        # y2s : sum.
    
        x1s, x2s, y1s, y2s = batch
        x1s, x2s, y1s, y2s = x1s.to(device), x2s.to(device), y1s.to(device), y2s.to(device)
        x2s = x2s.squeeze()

        # predictions
        predicted_y1s, predicted_y2s = network(x1s, x2s)

        # calculate the loss for image prediction.
        loss1 = F.cross_entropy(predicted_y1s, y1s)
        # calculate the loss for sum.
        loss2 = F.cross_entropy(predicted_y2s, y2s)
        loss = loss1 + loss2 # total loss.

        optimizer.zero_grad()           
        loss.backward()                 # calculate gradients
        optimizer.step()                # update weights 


        total_loss += loss.item()
        total_correct_mnist += get_num_correct(predicted_y1s, y1s)
        total_correct_sum += get_num_correct(predicted_y2s, y2s)
        #break
    #break
    image_accuracy = (total_correct_mnist/60000.0)*100;
    image_sum = (total_correct_sum/60000.0)*100;

    print(
        "epoch:", epoch,
        "batch_size: ", 100,
        "correct_images_count", total_correct_mnist,
        "correct_sum_count", total_correct_sum,
        "image_accuracy",image_accuracy, 
        "sum_accuracy", image_sum,
        "total_loss", total_loss
    )
```

The logs during training

```
epoch: 0 batch_size:  100 correct_images_count 54612 correct_sum_count 38350 image_accuracy 91.02 sum_accuracy 63.916666666666664 total_loss 968.1364297270775
epoch: 1 batch_size:  100 correct_images_count 57231 correct_sum_count 43314 image_accuracy 95.38499999999999 sum_accuracy 72.19 total_loss 692.24047935009
epoch: 2 batch_size:  100 correct_images_count 57431 correct_sum_count 43342 image_accuracy 95.71833333333333 sum_accuracy 72.23666666666666 total_loss 678.8417452573776
epoch: 3 batch_size:  100 correct_images_count 57670 correct_sum_count 43670 image_accuracy 96.11666666666666 sum_accuracy 72.78333333333333 total_loss 660.866904437542
epoch: 4 batch_size:  100 correct_images_count 57632 correct_sum_count 43569 image_accuracy 96.05333333333334 sum_accuracy 72.615 total_loss 666.2280558943748
epoch: 5 batch_size:  100 correct_images_count 57715 correct_sum_count 43771 image_accuracy 96.19166666666666 sum_accuracy 72.95166666666667 total_loss 656.60302901268
epoch: 6 batch_size:  100 correct_images_count 57567 correct_sum_count 43582 image_accuracy 95.94500000000001 sum_accuracy 72.63666666666667 total_loss 675.5272351503372
epoch: 7 batch_size:  100 correct_images_count 57889 correct_sum_count 43887 image_accuracy 96.48166666666667 sum_accuracy 73.14500000000001 total_loss 648.0555792152882
epoch: 8 batch_size:  100 correct_images_count 57799 correct_sum_count 43722 image_accuracy 96.33166666666668 sum_accuracy 72.87 total_loss 659.6569232940674
epoch: 9 batch_size:  100 correct_images_count 57940 correct_sum_count 43852 image_accuracy 96.56666666666666 sum_accuracy 73.08666666666667 total_loss 646.2071326375008

```
### Loss Function
Cross Entropy loss has been used for both are classification tasks. This loss function combines logsoftmax and nll loss. Hence, good for classification.
  - Image Classification.
  - The sum classification. This is classification because of the network. We have 19 features for every sum value. 


### Evaluating the Model.
Evaluation is done by using MNIST test data and predicting these on the trained model.
The custom data is generated similarly as training data.

```
# download test mnist data which consists of 10000 images.

test_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=False
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# test data loaders
mnist_adder_test_data = MNISTAdderDataset(test_set)
mnist_adder_test_data_loader = torch.utils.data.DataLoader(mnist_adder_test_data,
                                               batch_size=100,
                                               shuffle=False,
                                               )
```

Now, we predict the values of test data using the network model trained above.

```
total_image_correct = 0
total_sum_correct = 0

for batch in mnist_adder_test_data_loader:
  x1s, x2s, y1s, y2s = batch
  predicted_y1s, predicted_y2s = network(x1s, x2s)
  total_image_correct += get_num_correct(predicted_y1s, y1s)
  total_sum_correct += get_num_correct(predicted_y2s, y2s)

print("total_image_correct", total_image_correct, "image prediction accuracy:", (total_image_correct/10000.0) * 100)
print("total_sum_correct", total_sum_correct, "sum prediction accuracy", (total_sum_correct/10000) * 100)
```

The following accuracy has been achieved from the model.
```
total_image_correct 9625 image prediction accuracy: 96.25
total_sum_correct 7217 sum prediction accuracy 72.17
```
