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

#### DataSet Creation, Data Preprocessing, Data Augmentation
  
  The stanford sentiment dataset contains sentiment label for phrases. In order to get the sentiment of each sentence, we need to parse the parse tree. pytreebank<add link> package does the job effectively.

  pytreebank parses and computes the sentiment of each sentence. It returns label based on the following table.
  | raw sentiment label | mapped sentiment label | Description |
  |---------------------|------------------------|-------------|
  | [0, 0.2] | 0 | very negative |
  | [0.2, 0.4] | 1 |negative|
  | [0.4, 0.6] | 2 | neutral|
  | [0.6, 0.8] | 3 | very positive|
  | [0.8, 1.0] | 4 |positive|
