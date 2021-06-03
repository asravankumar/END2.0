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

In order to get the sentiment of each sentence, we need to parse the parse tree. pytreebank package does the job effectively.

The sentiment label are mapped to sentiment classes using the following cut-offs:
[0, 0.2] => very negative.
[0.2, 0.4] => negative.
[0.4, 0.6] => neutral.
[0.6, 0.8] => very positive.
[0.8, 1.0] => positive.

We download the sentences and sentiments from pytreebank and build models. 
