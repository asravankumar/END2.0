## Assignment 5 - Sentiment Analysis on Stanford Treebank Dataset

### Problem Statement

Write a neural network(LSTM) to predict the sentiment using Stanford Sentiment Analysis Dataset.
  - Use "Back Translate", "random_swap" and "random_delete" to augment the data.
  - Train the model and achieve 65%+ validation/text accuracy.

### Stanford Sentiment Analysis Dataset.
  - The stanford sentiment treebank dataset consists of fine-grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences.
  - The sentiment labels are between 0 to 1.
  - The dataset is available in two formats.
    - raw csv files.
    - Penn Treebank(PTB tree) format.
  - The raw csv files contains list of files which consists of list of 
    - sentences
    - phrases
    - trees mapping sentences with phrases
    - sentiment of each phrase.
  - The PTB format consists of the sentences, and the respective phrases in tree format.

### Proposed Solution

#### DataSet Download and Understanding
  
  The stanford sentiment dataset contains sentiment label for phrases. In order to get the sentiment of each sentence, we need to parse the parse tree. pytreebank<add link> package does the job effectively.

  pytreebank parses and computes the sentiment of each sentence. It returns label based on the following table.
  | raw sentiment label | mapped sentiment label | Description |
  |---------------------|------------------------|-------------|
  | [0, 0.2] | 0 | very negative |
  | [0.2, 0.4] | 1 |negative|
  | [0.4, 0.6] | 2 | neutral|
  | [0.6, 0.8] | 3 | positive|
  | [0.8, 1.0] | 4 |very positive|


  The dataset contains of three sets train, dev and test sets.
  We consider train and dev sets for our model training and testing respectively.

  Overall there are around 8544 training labelled sentences. The distribution of each label across can be visualized as follows.
  
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/value_counts.png)


  Clearly, we can see there is an imbalance for class labels 0 and 4 w.r.t the other labels. Training the model with such dataset would not result in accurate models.
  Data Augmentation techniques can be used to address this issue.

#### Data Augmentation
  We can apply data augmentation techniques like 'Back Translate' to increase the count of label 0 and 4.
  - Back Translate
    - In this technique, we translate the sentence to a random language and back translate it to english. The sentence may not be necessarily be the same. And hence, can be used to reduce the imbalance.

  Other techniques like 'random delete' and 'random swap' can be used to reduce the overfitting.
  - random delete
    - In this technique, randomly delete some words from sentences.
  - random swap
    - In this technique, randomly swap the positions of two words. do this n times.

  These three techniques are applied to improve the data quality of the dataset.
  The 'back translate' is applied on 0 and 4.
  The 'random delete' and 'random swap' are applied on 1, 2, 3. 
  After applying, following is the data distribution.

  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/value_counts_after_aug.png)


#### Data Preprocessing.
  The steps includes preprocessing all the sentences and creating torchtext Dataset objects for the model to use. 

  The following preprocessing have been performed:
  - lower case all sentences.
  - remove stopwords.
  - tokenize using spacy tokenizer.

####  Model And Training
  The network consists of the following layers.
  - Embedding layer with 300 dimensions
  - The embedding tokens are passed to 2-layer LSTM.
    - input nodes: 300
    - output nodes: 100
    - number of layers: 2
    - dropout: 0.2
  - Fully connected layer
    - input nodes: 100
    - output nodes: 5

  Network:
```
classifier(
  (embedding): Embedding(8002, 300)
  (encoder): LSTM(300, 100, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Linear(in_features=100, out_features=5, bias=True)
)
The model has 2,642,705 trainable parameters
```

  Adam Optimizer with crossentropyloss function is used.


Training Logs:

```
	Train Loss: 1.360 | Train Acc: 56.27%
	 Val. Loss: 1.529 |  Val. Acc: 33.96% 

	Train Loss: 1.305 | Train Acc: 63.72%
	 Val. Loss: 1.536 |  Val. Acc: 33.63% 

	Train Loss: 1.257 | Train Acc: 68.48%
	 Val. Loss: 1.538 |  Val. Acc: 34.26% 

	Train Loss: 1.213 | Train Acc: 72.54%
	 Val. Loss: 1.530 |  Val. Acc: 34.76% 

	Train Loss: 1.175 | Train Acc: 76.01%
	 Val. Loss: 1.528 |  Val. Acc: 35.39% 

	Train Loss: 1.144 | Train Acc: 79.01%
	 Val. Loss: 1.542 |  Val. Acc: 34.61% 

	Train Loss: 1.119 | Train Acc: 80.97%
	 Val. Loss: 1.539 |  Val. Acc: 33.90% 

	Train Loss: 1.098 | Train Acc: 82.56%
	 Val. Loss: 1.543 |  Val. Acc: 33.90% 

	Train Loss: 1.083 | Train Acc: 83.86%
	 Val. Loss: 1.544 |  Val. Acc: 33.93% 

	Train Loss: 1.071 | Train Acc: 84.91%
	 Val. Loss: 1.551 |  Val. Acc: 32.56% 

	Train Loss: 1.058 | Train Acc: 85.92%
	 Val. Loss: 1.548 |  Val. Acc: 33.63% 

	Train Loss: 1.048 | Train Acc: 86.67%
	 Val. Loss: 1.551 |  Val. Acc: 33.72% 

	Train Loss: 1.041 | Train Acc: 87.27%
	 Val. Loss: 1.561 |  Val. Acc: 32.56% 

	Train Loss: 1.035 | Train Acc: 87.63%
	 Val. Loss: 1.553 |  Val. Acc: 33.18% 

	Train Loss: 1.028 | Train Acc: 88.18%
	 Val. Loss: 1.556 |  Val. Acc: 32.56% 

	Train Loss: 1.026 | Train Acc: 88.37%
	 Val. Loss: 1.553 |  Val. Acc: 32.74% 

	Train Loss: 1.022 | Train Acc: 88.62%
	 Val. Loss: 1.556 |  Val. Acc: 32.89% 

	Train Loss: 1.023 | Train Acc: 88.65%
	 Val. Loss: 1.560 |  Val. Acc: 32.23% 

	Train Loss: 1.019 | Train Acc: 88.96%
	 Val. Loss: 1.554 |  Val. Acc: 33.42% 

	Train Loss: 1.018 | Train Acc: 88.98%
	 Val. Loss: 1.562 |  Val. Acc: 32.29% 

	Train Loss: 1.014 | Train Acc: 89.30%
	 Val. Loss: 1.566 |  Val. Acc: 32.38% 

	Train Loss: 1.010 | Train Acc: 89.62%
	 Val. Loss: 1.563 |  Val. Acc: 32.56% 

	Train Loss: 1.008 | Train Acc: 89.81%
	 Val. Loss: 1.566 |  Val. Acc: 33.10% 

	Train Loss: 1.008 | Train Acc: 89.86%
	 Val. Loss: 1.563 |  Val. Acc: 32.65% 

	Train Loss: 1.007 | Train Acc: 89.88%
	 Val. Loss: 1.562 |  Val. Acc: 32.83% 

	Train Loss: 1.008 | Train Acc: 89.89%
	 Val. Loss: 1.568 |  Val. Acc: 31.67% 

	Train Loss: 1.005 | Train Acc: 90.04%
	 Val. Loss: 1.556 |  Val. Acc: 33.10% 

	Train Loss: 1.003 | Train Acc: 90.17%
	 Val. Loss: 1.566 |  Val. Acc: 31.13% 

	Train Loss: 1.001 | Train Acc: 90.38%
	 Val. Loss: 1.559 |  Val. Acc: 33.10% 

	Train Loss: 1.000 | Train Acc: 90.46%
	 Val. Loss: 1.552 |  Val. Acc: 33.87% 

	Train Loss: 1.000 | Train Acc: 90.47%
	 Val. Loss: 1.547 |  Val. Acc: 34.23% 

	Train Loss: 1.001 | Train Acc: 90.49%
	 Val. Loss: 1.562 |  Val. Acc: 32.74% 

	Train Loss: 1.000 | Train Acc: 90.54%
	 Val. Loss: 1.560 |  Val. Acc: 33.60% 

	Train Loss: 0.997 | Train Acc: 90.73%
	 Val. Loss: 1.567 |  Val. Acc: 31.79% 

	Train Loss: 0.997 | Train Acc: 90.72%
	 Val. Loss: 1.566 |  Val. Acc: 32.05% 

	Train Loss: 0.997 | Train Acc: 90.82%
	 Val. Loss: 1.562 |  Val. Acc: 33.36% 

	Train Loss: 0.996 | Train Acc: 90.89%
	 Val. Loss: 1.567 |  Val. Acc: 32.50% 

	Train Loss: 0.995 | Train Acc: 90.99%
	 Val. Loss: 1.556 |  Val. Acc: 33.27% 

	Train Loss: 0.994 | Train Acc: 91.05%
	 Val. Loss: 1.565 |  Val. Acc: 32.65% 

	Train Loss: 0.993 | Train Acc: 91.16%
	 Val. Loss: 1.565 |  Val. Acc: 32.38% 

	Train Loss: 0.991 | Train Acc: 91.24%
	 Val. Loss: 1.563 |  Val. Acc: 32.56% 

	Train Loss: 0.991 | Train Acc: 91.27%
	 Val. Loss: 1.579 |  Val. Acc: 30.95% 

	Train Loss: 0.993 | Train Acc: 91.16%
	 Val. Loss: 1.550 |  Val. Acc: 33.78% 

	Train Loss: 0.992 | Train Acc: 91.24%
	 Val. Loss: 1.554 |  Val. Acc: 33.45% 

	Train Loss: 0.992 | Train Acc: 91.21%
	 Val. Loss: 1.571 |  Val. Acc: 31.99% 

	Train Loss: 0.993 | Train Acc: 91.14%
	 Val. Loss: 1.549 |  Val. Acc: 34.23% 

	Train Loss: 0.993 | Train Acc: 91.14%
	 Val. Loss: 1.554 |  Val. Acc: 33.69% 

	Train Loss: 0.991 | Train Acc: 91.37%
	 Val. Loss: 1.552 |  Val. Acc: 34.32% 

	Train Loss: 0.989 | Train Acc: 91.44%
	 Val. Loss: 1.557 |  Val. Acc: 33.48% 

	Train Loss: 0.988 | Train Acc: 91.53%
	 Val. Loss: 1.558 |  Val. Acc: 33.07% 

```

#### Results
  An Validation Accuracy of 33.07% was achieved after 50 epochs.

  
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/training_loss.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/valid_loss.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/training_accuracies.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/validation_accuracies.png)

#### Outcomes

- True Positives from validation Data. One example from each class.

```
sentence: It 's also , clearly , great fun .
label: 4
Predicted Label:  4 Very Positive

sentence: As unseemly as its title suggests .
label: 3
Predicted Label:  3 Positive

sentence: Smith is careful not to make fun of these curious owners of architectural oddities .
label: 2
Predicted Label:  2 Neutral

sentence: The overall effect is less like a children 's movie than a recruitment film for future Hollywood sellouts .
label: 1
Predicted Label:  1 Negative

sentence: It 's difficult to imagine the process that produced such a script , but here 's guessing that spray cheese and underarm noises played a crucial role .
label: 0
Predicted Label:  0 Very Negative

```

- False Positives. - sentiment positives predicted as negatives.

```
sentence: As a first-time director , Paxton has tapped something in himself as an actor that provides Frailty with its dark soul .
label: 3
Predicted Label:  0 Very Negative


sentence: By the time we learn that Andrew 's Turnabout Is Fair Play is every bit as awful as Borchardt 's Coven , we can enjoy it anyway .
label: 2
Predicted Label:  1 Negative

sentence: There 's something auspicious , and daring , too , about the artistic instinct that pushes a majority-oriented director like Steven Spielberg to follow A.I. with this challenging report so liable to unnerve the majority .
label: 3
Predicted Label:  1 Negative

```

- False Positives. sentiment negatives predicted as positives.

```
sentence: Chabrol has taken promising material for a black comedy and turned it instead into a somber chamber drama .
label: 1
Predicted Label:  3 Positive

```

#### Future Enhancements
- Use better data augmentation and preprocessing techniques to enhance the data quality.
- Train using bi-LSTM or GRU.

