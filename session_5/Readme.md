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
  | [0.6, 0.8] | 3 | very positive|
  | [0.8, 1.0] | 4 |positive|


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
  Train Loss: 1.593 | Train Acc: 25.56%
	 Val. Loss: 1.579 |  Val. Acc: 27.71% 

	Train Loss: 1.582 | Train Acc: 29.12%
	 Val. Loss: 1.570 |  Val. Acc: 29.35% 

	Train Loss: 1.552 | Train Acc: 34.43%
	 Val. Loss: 1.544 |  Val. Acc: 33.96% 

	Train Loss: 1.500 | Train Acc: 39.29%
	 Val. Loss: 1.541 |  Val. Acc: 34.35% 

	Train Loss: 1.450 | Train Acc: 45.25%
	 Val. Loss: 1.535 |  Val. Acc: 35.36% 

	Train Loss: 1.399 | Train Acc: 51.61%
	 Val. Loss: 1.527 |  Val. Acc: 35.42% 

	Train Loss: 1.346 | Train Acc: 58.07%
	 Val. Loss: 1.537 |  Val. Acc: 34.40% 

	Train Loss: 1.295 | Train Acc: 64.05%
	 Val. Loss: 1.535 |  Val. Acc: 34.88% 

	Train Loss: 1.246 | Train Acc: 69.25%
	 Val. Loss: 1.544 |  Val. Acc: 33.12% 

	Train Loss: 1.203 | Train Acc: 73.62%
	 Val. Loss: 1.549 |  Val. Acc: 32.68% 

	Train Loss: 1.166 | Train Acc: 77.15%
	 Val. Loss: 1.546 |  Val. Acc: 33.57% 

	Train Loss: 1.136 | Train Acc: 80.05%
	 Val. Loss: 1.547 |  Val. Acc: 33.63% 

	Train Loss: 1.111 | Train Acc: 81.84%
	 Val. Loss: 1.558 |  Val. Acc: 32.44% 

	Train Loss: 1.092 | Train Acc: 83.41%
	 Val. Loss: 1.551 |  Val. Acc: 33.51% 

	Train Loss: 1.076 | Train Acc: 84.75%
	 Val. Loss: 1.557 |  Val. Acc: 32.80% 

	Train Loss: 1.064 | Train Acc: 85.58%
	 Val. Loss: 1.560 |  Val. Acc: 31.70% 

	Train Loss: 1.054 | Train Acc: 86.30%
	 Val. Loss: 1.562 |  Val. Acc: 32.38% 

	Train Loss: 1.047 | Train Acc: 86.81%
	 Val. Loss: 1.563 |  Val. Acc: 31.79% 

	Train Loss: 1.040 | Train Acc: 87.22%
	 Val. Loss: 1.563 |  Val. Acc: 32.35% 

	Train Loss: 1.035 | Train Acc: 87.58%
	 Val. Loss: 1.560 |  Val. Acc: 33.10% 
```

#### Results
  An Validation Accuracy of 33.10% was achieved after 20 epochs.

  
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/training_loss.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/validation_loss.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/training_accuracy.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/validation_accuracy.png)

#### Outcomes
```
```
