## Assignment 6 - Sentiment Analysis on Tweets using Encoder Decoder Architecture

### Problem Statement
Using Encoder Decoder Architecture, perform sentiment analysis on tweets dataset.
  - encoder: an RNN/LSTM layer takes the words in a sentence one by one and finally converts them into a single vector. VERY IMPORTANT TO MAKE THIS SINGLE VECTOR this single vector is then sent to another RNN/LSTM that also takes the last prediction as its second input. Then we take the final vector from this Cell and send this final vector to a Linear Layer and make the final prediction. 
  - This is how it will look:
    - embedding
    - word from a sentence +last hidden vector -> encoder -> single vector
    - single vector + last hidden vector -> decoder -> single vector
    - single vector -> FC layer -> Prediction

#### DataSet
  - Dataset consists of 1364 labelled tweets with three sentiments :- negative, neutral & positive.
  - The labels are as follows.
    - 0: Negative
    - 1: Neutral
    - 2: Positive

####  Model And Training
  The network consists of the following:
  - Encoder
  - Decoder
  - Fully Connected Layer

  Network:
```
EncoderDecoderClassifier(
  (encoder): Encoder(
    (embedding): Embedding(4651, 300)
    (rnn_layer): RNN(300, 100, batch_first=True)
  )
  (decoder): Decoder(
    (decoder): LSTM(100, 100, batch_first=True)
    (decoder2): LSTM(100, 100, batch_first=True)
  )
  (fc): Linear(in_features=100, out_features=3, bias=True)
)
The model has 1,597,403 trainable parameters
```

#### Encoder Decoder Classifier Classes

  - Encoder

```
class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    # Encoder class which accepts a sequence(tweet) and converts it into a context vector.
    super().__init__()

    # Embedding layer
    self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # RNN Layer for encoding
    self.rnn_layer = nn.RNN(embedding_dim,
                          hidden_dim,
                          batch_first=True)
    
  def forward(self, text, text_lengths):
    # text = [batch size, sent_length]
    embedded_text         = self.embedding(text)

    # embedded = [batch size, sent_len, emb dim]

    # packed sequence
    packed_embedded_text  = nn.utils.rnn.pack_padded_sequence(embedded_text,
                                                        text_lengths.cpu(),
                                                        batch_first=True)
    # input sequence to the rnn layer 
    encoder_output, hidden   = self.rnn_layer(packed_embedded_text)
    # rnn output in packed format which contains outputs at every sequence.
    # the hidden will be of last state only.
    # hidden = [1 , batch_size, hidden_dim]
    # Note that, the hidden tensor will not be in the batch_first = True shape. Only the output tensor will be in batch_first if it is set to true.

    # unpack the encoder rnn output 
    encoder_output, encoder_output_lengths = nn.utils.rnn.pad_packed_sequence(encoder_output, batch_first=True)
    # will be in batch_first = True shape
    # encoder_output = [batch_size, sent_len, hidden_dim]

    # here returning the output at all states.
    # and the last hidden vector which is the SINGLE context vector for the input sequence. 
    return(encoder_output, hidden)
```

  - Decoder

```
class Decoder(nn.Module):
  def __init__(self, encoder_output_dim, hidden_dim):
    super().__init__()

    # lstm layer as part of decoder.
    # the encoder emits a hidden vector with one sequence. Hence, lstm is here for once in the pipeline.

    self.decoder = nn.LSTM(encoder_output_dim, 
                       hidden_dim,  
                       batch_first=True)
    self.decoder2 = nn.LSTM(hidden_dim, 
                       hidden_dim,  
                       batch_first=True)

  def forward(self, single_vector):
    # Here single_vector is the hidden vector from the encoder.
    # single_vector = [batch_size, 1, hidden_dim]

    # encoder_output = encoder_output.squeeze(0).unsqueeze(1)
    # encoder_output
    output, (hidden_vector, cell_vector) = self.decoder(single_vector)

    # hidden_vector is the decoded vector which is input to the fully connected layer.
    # hidden_vector = [1, batch_size, hidden_dim]
    output2, (hidden_vector2, cell_vector2) = self.decoder(output, (hidden_vector, cell_vector))

    return(output2, hidden_vector2)
```

  - EncoderDecoderClassifier

```
class EncoderDecoderClassifier(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(EncoderDecoderClassifier, self).__init__()

    self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    self.decoder = Decoder(hidden_dim, hidden_dim)
    self.fc      = nn.Linear(hidden_dim, output_dim)

  def forward(self, text, text_lengths):
    # Encoder encodes using an rnn.
    encoder_output, encoder_hidden = self.encoder(text, text_lengths)

    # the hidden vector emitted by rnn is not in batch_first=True shape. Hence converting.
    # encoder_hidden = [1, batch_size, hidden_dim]
    encoder_hidden = encoder_hidden.squeeze(0).unsqueeze(1)
    # After reshaping
    # encoder_hidden = [batch_size, 1, hidden_dim]
    
    # Decode the input encoded single vector.
    decoder_output, decoder_hidden = self.decoder(encoder_hidden)

    # the hidden vector emitted by the decoder is in the batch_first=False shape. Hence convert it to shape for linear layer.
    decoder_hidden = decoder_hidden.squeeze(0) #.unsqueeze(1)
    # input to fully connected layer dimension = [batch_size, hidden_dim]
    dense_outputs = self.fc(decoder_hidden)

    # fully connected layer output = [batch_size, output_dim]
    final_output = F.softmax(dense_outputs, dim=0)
    return(final_output, encoder_output, encoder_hidden, decoder_output)
```

Adam Optimizer with crossentropyloss function is used.


Training Logs:

```
Train Loss: 1.095 | Train Acc: 50.34%
	 Val. Loss: 1.092 |  Val. Acc: 57.56% 

	Train Loss: 1.071 | Train Acc: 60.43%
	 Val. Loss: 1.055 |  Val. Acc: 60.98% 

	Train Loss: 1.027 | Train Acc: 66.10%
	 Val. Loss: 1.037 |  Val. Acc: 60.49% 

	Train Loss: 1.003 | Train Acc: 70.93%
	 Val. Loss: 1.040 |  Val. Acc: 61.46% 

	Train Loss: 0.981 | Train Acc: 72.16%
	 Val. Loss: 1.041 |  Val. Acc: 57.07% 

	Train Loss: 0.978 | Train Acc: 74.63%
	 Val. Loss: 1.035 |  Val. Acc: 59.02% 

	Train Loss: 0.958 | Train Acc: 78.86%
	 Val. Loss: 1.031 |  Val. Acc: 62.93% 

	Train Loss: 0.955 | Train Acc: 79.74%
	 Val. Loss: 1.033 |  Val. Acc: 61.95% 

	Train Loss: 0.953 | Train Acc: 78.10%
	 Val. Loss: 1.035 |  Val. Acc: 64.39% 

	Train Loss: 0.946 | Train Acc: 80.09%
	 Val. Loss: 1.033 |  Val. Acc: 63.90% 

	Train Loss: 0.939 | Train Acc: 82.84%
	 Val. Loss: 1.031 |  Val. Acc: 63.90% 

	Train Loss: 0.939 | Train Acc: 81.12%
	 Val. Loss: 1.028 |  Val. Acc: 64.88% 

	Train Loss: 0.938 | Train Acc: 83.88%
	 Val. Loss: 1.030 |  Val. Acc: 66.34% 

	Train Loss: 0.937 | Train Acc: 82.13%
	 Val. Loss: 1.027 |  Val. Acc: 67.32% 

	Train Loss: 0.935 | Train Acc: 84.48%
	 Val. Loss: 1.030 |  Val. Acc: 66.83% 

	Train Loss: 0.930 | Train Acc: 87.41%
	 Val. Loss: 1.029 |  Val. Acc: 66.83% 

	Train Loss: 0.932 | Train Acc: 84.09%
	 Val. Loss: 1.029 |  Val. Acc: 68.78% 

	Train Loss: 0.935 | Train Acc: 84.57%
	 Val. Loss: 1.029 |  Val. Acc: 68.78% 

	Train Loss: 0.938 | Train Acc: 84.72%
	 Val. Loss: 1.027 |  Val. Acc: 67.80% 

	Train Loss: 0.931 | Train Acc: 85.43%
	 Val. Loss: 1.026 |  Val. Acc: 68.29% 

	Train Loss: 0.936 | Train Acc: 83.19%
	 Val. Loss: 1.031 |  Val. Acc: 67.32% 

	Train Loss: 0.935 | Train Acc: 84.48%
	 Val. Loss: 1.030 |  Val. Acc: 67.32% 

	Train Loss: 0.932 | Train Acc: 87.50%
	 Val. Loss: 1.026 |  Val. Acc: 68.78% 

	Train Loss: 0.929 | Train Acc: 88.53%
	 Val. Loss: 1.025 |  Val. Acc: 69.27% 

	Train Loss: 0.928 | Train Acc: 86.64%
	 Val. Loss: 1.026 |  Val. Acc: 69.27% 

	Train Loss: 0.940 | Train Acc: 83.79%
	 Val. Loss: 1.023 |  Val. Acc: 68.78% 

	Train Loss: 0.938 | Train Acc: 86.21%
	 Val. Loss: 1.022 |  Val. Acc: 69.76% 

	Train Loss: 0.930 | Train Acc: 86.12%
	 Val. Loss: 1.023 |  Val. Acc: 70.73% 

	Train Loss: 0.937 | Train Acc: 85.78%
	 Val. Loss: 1.025 |  Val. Acc: 70.73% 

	Train Loss: 0.933 | Train Acc: 84.91%
	 Val. Loss: 1.015 |  Val. Acc: 70.24% 

	Train Loss: 0.937 | Train Acc: 83.73%
	 Val. Loss: 1.018 |  Val. Acc: 69.27% 

	Train Loss: 0.949 | Train Acc: 81.55%
	 Val. Loss: 1.019 |  Val. Acc: 67.32% 

	Train Loss: 0.930 | Train Acc: 85.43%
	 Val. Loss: 1.016 |  Val. Acc: 67.32% 

	Train Loss: 0.930 | Train Acc: 87.67%
	 Val. Loss: 1.016 |  Val. Acc: 68.29% 

	Train Loss: 0.930 | Train Acc: 87.31%
	 Val. Loss: 1.017 |  Val. Acc: 68.78% 

	Train Loss: 0.922 | Train Acc: 89.05%
	 Val. Loss: 1.017 |  Val. Acc: 68.78% 

	Train Loss: 0.933 | Train Acc: 87.76%
	 Val. Loss: 1.015 |  Val. Acc: 69.27% 

	Train Loss: 0.927 | Train Acc: 88.73%
	 Val. Loss: 1.015 |  Val. Acc: 69.27% 

	Train Loss: 0.930 | Train Acc: 86.72%
	 Val. Loss: 1.016 |  Val. Acc: 69.76% 

	Train Loss: 0.932 | Train Acc: 86.90%
	 Val. Loss: 1.017 |  Val. Acc: 70.24% 

	Train Loss: 0.932 | Train Acc: 85.43%
	 Val. Loss: 1.017 |  Val. Acc: 71.71% 

	Train Loss: 0.932 | Train Acc: 84.66%
	 Val. Loss: 1.019 |  Val. Acc: 71.22% 

	Train Loss: 0.929 | Train Acc: 86.68%
	 Val. Loss: 1.018 |  Val. Acc: 71.22% 

	Train Loss: 0.928 | Train Acc: 87.76%
	 Val. Loss: 1.019 |  Val. Acc: 71.22% 

	Train Loss: 0.934 | Train Acc: 85.78%
	 Val. Loss: 1.018 |  Val. Acc: 71.71% 

	Train Loss: 0.927 | Train Acc: 87.93%
	 Val. Loss: 1.021 |  Val. Acc: 69.76% 

	Train Loss: 0.936 | Train Acc: 85.43%
	 Val. Loss: 1.025 |  Val. Acc: 68.78% 

	Train Loss: 0.935 | Train Acc: 84.81%
	 Val. Loss: 1.023 |  Val. Acc: 68.29% 

	Train Loss: 0.929 | Train Acc: 85.95%
	 Val. Loss: 1.014 |  Val. Acc: 69.76% 

	Train Loss: 0.933 | Train Acc: 85.60%
	 Val. Loss: 1.014 |  Val. Acc: 70.73% 
```

#### Results
  An Validation Accuracy of 33.07% was achieved after 50 epochs.

  
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_6/training_loss.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_6/valid_loss.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_6/training_accuracies.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_6/validation_accuracies.png)

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

