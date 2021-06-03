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
  After applying, following is the data distribution.

  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_5/value_counts_after_augmentation.png)
