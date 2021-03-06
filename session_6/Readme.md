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
    - 1: Positive
    - 2: Neutral

####  Model And Training
  The network consists of the following:
  - Encoder
    - Encoder is implemented by RNN.
    - The hidden vector of the last state is the single vector.
    - This single vector is passed to the decoder as input.
  - Decoder
    - Decoder is implemented by LSTM.
    - 2 lstms are used.
    - The input is the single vector which is the last hidden vector of the encoder RNN.
    - The first LSTM accepts the single vector which is of shape [batch_size, 1, hidden_dimension]
    - The output of this LSTM is sent as input to the 2nd LSTM.
    - The hidden and cell vector of the first LSTM is passed to the second one.
    - The hidden vector of the 2nd LSTM is then passed to a fully connected layer.
  - Fully Connected Layer
    - A fully connected layer with hidden_dimension as input layer nodes and three as output nodes(one for each label).

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

  This [playbook](https://github.com/asravankumar/END2.0/blob/master/session_6/encoder_decoder_tweets_sentiment_2_lstm_decoder.ipynb) is being used for the above architecture.

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

    # output will be in batch_first = True shape
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
    output2, (hidden_vector2, cell_vector2) = self.decoder2(output, (hidden_vector, cell_vector))

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
Train Loss: 1.095 | Train Acc: 50.50%
	 Val. Loss: 1.093 |  Val. Acc: 55.12% 

	Train Loss: 1.070 | Train Acc: 59.91%
	 Val. Loss: 1.065 |  Val. Acc: 58.05% 

	Train Loss: 1.029 | Train Acc: 65.06%
	 Val. Loss: 1.042 |  Val. Acc: 63.90% 

	Train Loss: 1.000 | Train Acc: 68.69%
	 Val. Loss: 1.037 |  Val. Acc: 64.88% 

	Train Loss: 0.985 | Train Acc: 73.00%
	 Val. Loss: 1.029 |  Val. Acc: 62.93% 

	Train Loss: 0.963 | Train Acc: 76.55%
	 Val. Loss: 1.029 |  Val. Acc: 64.88% 

	Train Loss: 0.963 | Train Acc: 76.96%
	 Val. Loss: 1.025 |  Val. Acc: 64.88% 

	Train Loss: 0.951 | Train Acc: 78.86%
	 Val. Loss: 1.018 |  Val. Acc: 68.78% 

	Train Loss: 0.951 | Train Acc: 80.17%
	 Val. Loss: 1.018 |  Val. Acc: 66.34% 

	Train Loss: 0.950 | Train Acc: 79.74%
	 Val. Loss: 1.017 |  Val. Acc: 64.88% 

	Train Loss: 0.942 | Train Acc: 80.34%
	 Val. Loss: 1.015 |  Val. Acc: 64.39% 

	Train Loss: 0.940 | Train Acc: 80.43%
	 Val. Loss: 1.014 |  Val. Acc: 65.37% 

	Train Loss: 0.938 | Train Acc: 82.84%
	 Val. Loss: 1.016 |  Val. Acc: 68.78% 

	Train Loss: 0.937 | Train Acc: 82.24%
	 Val. Loss: 1.015 |  Val. Acc: 67.32% 

	Train Loss: 0.938 | Train Acc: 81.92%
	 Val. Loss: 1.015 |  Val. Acc: 66.83% 

	Train Loss: 0.935 | Train Acc: 83.53%
	 Val. Loss: 1.015 |  Val. Acc: 67.80% 

	Train Loss: 0.934 | Train Acc: 84.14%
	 Val. Loss: 1.018 |  Val. Acc: 67.80% 

	Train Loss: 0.932 | Train Acc: 84.74%
	 Val. Loss: 1.017 |  Val. Acc: 65.85% 

	Train Loss: 0.935 | Train Acc: 83.62%
	 Val. Loss: 1.015 |  Val. Acc: 65.37% 

	Train Loss: 0.937 | Train Acc: 81.98%
	 Val. Loss: 1.015 |  Val. Acc: 69.76% 

	Train Loss: 0.931 | Train Acc: 84.31%
	 Val. Loss: 1.016 |  Val. Acc: 68.78% 

	Train Loss: 0.933 | Train Acc: 83.97%
	 Val. Loss: 1.013 |  Val. Acc: 66.83% 

	Train Loss: 0.934 | Train Acc: 82.41%
	 Val. Loss: 1.016 |  Val. Acc: 67.80% 

	Train Loss: 0.932 | Train Acc: 85.52%
	 Val. Loss: 1.016 |  Val. Acc: 67.80% 

	Train Loss: 0.929 | Train Acc: 85.26%
	 Val. Loss: 1.014 |  Val. Acc: 67.32% 

	Train Loss: 0.930 | Train Acc: 85.52%
	 Val. Loss: 1.022 |  Val. Acc: 66.34% 

	Train Loss: 0.939 | Train Acc: 84.05%
	 Val. Loss: 1.014 |  Val. Acc: 67.80% 

	Train Loss: 0.934 | Train Acc: 84.59%
	 Val. Loss: 1.015 |  Val. Acc: 68.29% 

	Train Loss: 0.926 | Train Acc: 86.29%
	 Val. Loss: 1.014 |  Val. Acc: 67.32% 

	Train Loss: 0.934 | Train Acc: 84.40%
	 Val. Loss: 1.013 |  Val. Acc: 67.80% 

	Train Loss: 0.930 | Train Acc: 85.67%
	 Val. Loss: 1.015 |  Val. Acc: 69.27% 

	Train Loss: 0.931 | Train Acc: 85.93%
	 Val. Loss: 1.014 |  Val. Acc: 69.27% 

	Train Loss: 0.942 | Train Acc: 83.71%
	 Val. Loss: 1.017 |  Val. Acc: 69.27% 

	Train Loss: 0.927 | Train Acc: 85.69%
	 Val. Loss: 1.015 |  Val. Acc: 69.27% 

	Train Loss: 0.927 | Train Acc: 86.72%
	 Val. Loss: 1.014 |  Val. Acc: 68.78% 

	Train Loss: 0.930 | Train Acc: 85.63%
	 Val. Loss: 1.017 |  Val. Acc: 67.32% 

	Train Loss: 0.923 | Train Acc: 88.53%
	 Val. Loss: 1.022 |  Val. Acc: 66.83% 

	Train Loss: 0.933 | Train Acc: 86.72%
	 Val. Loss: 1.018 |  Val. Acc: 69.76% 

	Train Loss: 0.927 | Train Acc: 88.17%
	 Val. Loss: 1.015 |  Val. Acc: 69.76% 

	Train Loss: 0.930 | Train Acc: 85.69%
	 Val. Loss: 1.017 |  Val. Acc: 68.78% 

	Train Loss: 0.931 | Train Acc: 86.47%
	 Val. Loss: 1.017 |  Val. Acc: 70.24% 

	Train Loss: 0.932 | Train Acc: 86.12%
	 Val. Loss: 1.021 |  Val. Acc: 70.24% 

	Train Loss: 0.931 | Train Acc: 84.22%
	 Val. Loss: 1.017 |  Val. Acc: 69.27% 

	Train Loss: 0.929 | Train Acc: 86.14%
	 Val. Loss: 1.018 |  Val. Acc: 69.27% 

	Train Loss: 0.927 | Train Acc: 87.24%
	 Val. Loss: 1.016 |  Val. Acc: 69.27% 

	Train Loss: 0.933 | Train Acc: 86.81%
	 Val. Loss: 1.017 |  Val. Acc: 68.78% 

	Train Loss: 0.926 | Train Acc: 86.29%
	 Val. Loss: 1.017 |  Val. Acc: 67.80% 

	Train Loss: 0.932 | Train Acc: 87.07%
	 Val. Loss: 1.017 |  Val. Acc: 68.29% 

	Train Loss: 0.929 | Train Acc: 87.13%
	 Val. Loss: 1.019 |  Val. Acc: 69.76% 

	Train Loss: 0.925 | Train Acc: 87.93%
	 Val. Loss: 1.016 |  Val. Acc: 70.24% 
```

#### Results
  An Validation Accuracy of 70.24% was achieved after 50 epochs.

  
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_6/training_loss.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_6/valid_loss.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_6/training_accuracies.png)
  ![alt text](https://github.com/asravankumar/END2.0/blob/master/session_6/validation_accuracies.png)

#### Intermediate Vectors for an example
  - Tweet: #edshow Whenever Obama tells the truth about the Gop, they boo hoo hoo and call him a bully.

  - Encoder Output after each token step:

```
encoding output vector: tensor([ 0.3184, -0.7582, -0.2498,  0.6358, -0.7074, -0.1639,  0.1268,  0.6927,
         0.3888,  0.9658,  0.0936, -0.9840, -0.3199,  0.2241,  0.2063, -0.9582,
         0.4742, -0.7685,  0.8296, -0.5967, -0.9885, -0.8501, -0.4127,  0.9805,
        -0.9958,  0.4343,  0.9408,  0.3124, -0.1117, -0.3498, -0.4366,  0.9794,
        -0.6898,  0.0073, -0.8230, -0.0485, -0.9717, -0.6958, -0.0212, -0.8098,
        -0.2424, -0.7672,  0.0356, -0.6456, -0.8653,  0.9758, -0.3249,  0.9766,
        -0.3139,  0.9506, -0.2566, -0.2896,  0.9861,  0.9350,  0.8717, -0.2761,
         0.7900, -0.9853,  0.8331,  0.5255,  0.8727, -0.7792, -0.3003, -0.7466,
        -0.0197,  0.7823,  0.9855,  0.6326,  0.2261,  0.2195,  0.8886, -0.7431,
         0.4201,  0.3087, -0.7448, -0.9645, -0.3026, -0.9567, -0.6131, -0.8406,
        -0.0513,  0.0021, -0.0968,  0.4702, -0.3302, -0.6279,  0.4357, -0.2746,
         0.7565, -0.5658, -0.2303,  0.9468, -0.9551,  0.9323, -0.5694,  0.0459,
        -0.9444,  0.9825, -0.0763,  0.0259], grad_fn=<SelectBackward>)
--------
input word: "edshow"
encoding output vector: tensor([-0.8306, -0.5043, -0.1851, -0.0990, -0.2999,  0.8665,  0.5230, -0.4688,
        -0.6646,  0.9997, -0.8474, -0.0414, -0.9589,  0.0865, -0.9489, -0.1233,
         0.3319,  0.8367, -0.5844, -0.9456, -0.8155,  0.0227,  0.5132,  0.0368,
        -0.7363,  0.6340, -0.7174, -0.7548, -0.8892, -0.8942,  0.4967,  0.3757,
         0.3959, -0.0350, -0.6567,  0.9331,  0.0449,  0.5228, -0.5532, -0.3492,
        -0.0661,  0.8383,  0.9003,  0.0351,  0.3269,  0.9707,  0.2669, -0.9608,
        -0.3588,  0.0430, -0.8151,  0.7553,  0.6129,  0.4784, -0.3945, -0.9849,
         0.1356,  0.0974,  0.7370, -0.6478, -0.5768, -0.5070, -0.8058,  0.8997,
         0.7501,  0.1182, -0.8883,  0.2193, -0.9348,  0.7256,  0.9044, -0.1682,
        -0.6951, -0.9707,  0.0422,  0.6848, -0.8821, -0.8783,  0.4124,  0.1646,
        -0.2524, -0.0484,  0.5404,  0.9464,  0.7615,  0.3351, -0.0588,  0.0639,
        -0.6468, -0.2705,  0.2784, -0.7064,  0.9917, -0.7101, -0.0574,  0.4337,
         0.6093, -0.3502, -0.9578, -0.9424], grad_fn=<SelectBackward>)
--------
input word: "Whenever"
encoding output vector: tensor([ 0.6559,  0.7830,  0.5614,  0.2445,  0.8968,  0.5322, -0.6595,  0.8170,
         0.7909,  0.9212, -0.8888,  0.9099,  0.9640,  0.0099,  0.3334, -0.4582,
        -0.6882,  0.6635,  0.7687,  0.5020, -0.2254, -0.0284, -0.6895,  0.6924,
        -0.1598, -0.0659,  0.7227, -0.5485,  0.8104, -0.7100,  0.9044,  0.8090,
         0.4478,  0.6642, -0.0496, -0.8677,  0.7189, -0.5532,  0.4282, -0.9195,
         0.7977, -0.9690, -0.6227,  0.6693,  0.0594,  0.8175, -0.0858,  0.1142,
         0.1081,  0.0879,  0.9942,  0.0781, -0.1041,  0.9797,  0.4590, -0.7467,
         0.3244, -0.3670,  0.0090, -0.3152,  0.9408, -0.0528,  0.7472, -0.5082,
         0.5027,  0.0012,  0.9860, -0.5679, -0.2302, -0.8796, -0.6701, -0.1659,
        -0.1763, -0.6690, -0.6925,  0.8972,  0.9504,  0.1119, -0.1574,  0.8312,
         0.7174, -0.9762,  0.1967, -0.9048, -0.7282, -0.6725,  0.1162, -0.7795,
         0.4736, -0.9551, -0.5507, -0.9005, -0.4985,  0.1986, -0.9562, -0.7022,
         0.9612,  0.0566,  0.0359,  0.2798], grad_fn=<SelectBackward>)
--------
input word: "Obama"
encoding output vector: tensor([-0.6597, -0.5297,  0.6739,  0.6817, -0.2149,  0.9592, -0.7333,  0.5559,
        -0.2361,  0.6949,  0.3537, -0.5116, -0.3275,  0.4605,  0.2148,  0.1050,
         0.2184, -0.1322, -0.8769, -0.3839, -0.9842,  0.6566, -0.7897,  0.8917,
        -0.5678,  0.9324, -0.8213,  0.9409, -0.9596, -0.7671,  0.6111,  0.9939,
        -0.7184, -0.8563, -0.4942,  0.3141, -0.0360,  0.9930,  0.5179,  0.5856,
        -0.9926, -0.2626,  0.3523, -0.4546,  0.9060,  0.6351,  0.9539,  0.6079,
         0.7690,  0.3033,  0.4290, -0.7904,  0.9386, -0.9656, -0.8174, -0.4192,
        -0.9753,  0.4245, -0.6078, -0.0232, -0.8041, -0.6203, -0.9934, -0.9790,
        -0.3392, -0.7867, -0.9565, -0.2182,  0.3395,  0.9455, -0.4738,  0.9955,
        -0.6663,  0.8140, -0.7217, -0.5114, -0.7301, -0.9686,  0.5728,  0.7771,
         0.8901, -0.8251,  0.7465,  0.8189,  0.2524,  0.9603, -0.2097,  0.7432,
        -0.4702,  0.7370,  0.8227, -0.9709, -0.9344, -0.4306, -0.3644,  0.7868,
        -0.8208, -0.7705,  0.9648, -0.1411], grad_fn=<SelectBackward>)
--------
input word: "tells"
encoding output vector: tensor([-0.3201,  0.8657,  0.1850,  0.3394,  0.3656, -0.0578, -0.9014,  0.8524,
         0.4747, -0.6184, -0.8432,  0.7700, -0.9575, -0.8547, -0.8175,  0.3529,
         0.8442, -0.8648, -0.8565, -0.7312,  0.6624,  0.8831, -0.8414, -0.4761,
         0.5543, -0.5178,  0.3185,  0.9268, -0.2350,  0.7543, -0.5193, -0.7261,
         0.7706,  0.3252,  0.9512, -0.8987, -0.6639, -0.7987,  0.1956, -0.8184,
         0.9778, -0.4975,  0.2883,  0.4125,  0.8933,  0.7473, -0.9235,  0.2140,
         0.9966, -0.6916,  0.8933, -0.9448,  0.8228, -0.5493, -0.8847, -0.3910,
        -0.7538,  0.5525,  0.4538,  0.2063,  0.9292,  0.2197,  0.9995, -0.9721,
        -0.0165,  0.2013,  0.6657, -0.7943, -0.1983,  0.0045,  0.2549, -0.7498,
        -0.7699, -0.4749, -0.9935, -0.1196,  0.9265,  0.1844, -0.7998,  0.9356,
        -0.7176, -0.7676,  0.2001,  0.8739,  0.7371, -0.8704, -0.7007, -0.9206,
        -0.7390, -0.9045,  0.8564, -0.9754, -0.7669,  0.9193, -0.9483, -0.9470,
        -0.7020, -0.7961, -0.8787, -0.6788], grad_fn=<SelectBackward>)
--------
input word: "the"
encoding output vector: tensor([ 0.2505, -0.9887,  0.6738, -0.0772, -0.1305,  0.7464,  0.0562,  0.6882,
         0.8657,  0.4217,  0.7573, -0.9037, -0.4160, -0.7028, -0.4443, -0.9585,
         0.4901, -0.9566, -0.8647,  0.9585, -0.6458, -0.2018,  0.9137,  0.9812,
        -0.8488, -0.9789, -0.8889,  0.4825,  0.4119, -0.2563,  0.1576, -0.6448,
        -0.8364, -0.9567,  0.1323,  0.0478, -0.5905, -0.3386, -0.6611,  0.9740,
        -0.6189,  0.6085, -0.2019, -0.7092,  0.7481,  0.8973, -0.5715,  0.5893,
        -0.4429, -0.7157, -0.4215,  0.9029, -0.3243, -0.4496, -0.8475,  0.4444,
         0.0326, -0.6826, -0.4375, -0.2917, -0.4443, -0.8100,  0.3440, -0.7680,
         0.9760, -0.4948, -0.8454,  0.9282,  0.9482, -0.3172,  0.1869,  0.6915,
         0.5994, -0.4112, -0.4731, -0.5971, -0.8733, -0.3378, -0.9395,  0.9669,
        -0.0707, -0.2505, -0.5830, -0.8829,  0.4805,  0.5256, -0.0886,  0.7738,
        -0.1938,  0.5331,  0.8069,  0.9942, -0.6355, -0.1331, -0.4263, -0.0700,
        -0.8613, -0.5379, -0.0122, -0.2139], grad_fn=<SelectBackward>)
--------
input word: "truth"
encoding output vector: tensor([ 0.3659,  0.9676,  0.9973, -0.9664,  0.8233, -0.8909,  0.8927,  0.9139,
         0.9463,  0.9806, -0.9892, -0.4878, -0.5620,  0.7691,  0.9882, -0.5579,
         0.9632, -0.9591,  0.7407, -0.4443, -0.7682, -0.9724,  0.2627, -0.9523,
        -0.1065,  0.3727, -0.2913, -0.0306,  0.8472, -0.9275,  0.4599,  0.4216,
        -0.9283,  0.9815,  0.6304,  0.7297,  0.7448, -0.4484, -0.7395,  0.8207,
         0.9707, -0.6319,  0.7909,  0.6782, -0.9695,  0.9666,  0.9751,  0.7678,
        -0.4999, -0.2198, -0.9065,  0.2581, -0.9174, -0.1289,  0.4447,  0.1668,
         0.9694, -0.5314,  0.4025,  0.9784,  0.6879,  0.8746,  0.9932, -0.2553,
        -0.5685, -0.5357,  0.9661, -0.8543, -0.8436,  0.7029, -0.3665, -0.7044,
        -0.9336,  0.0145, -0.6044,  0.0185, -0.1853,  0.9846, -0.7679, -0.2292,
         0.1943,  0.5848,  0.9813,  0.9095,  0.8550, -0.8038, -0.2873,  0.7404,
         0.6167, -0.9462, -0.9179,  0.1146,  0.9238, -0.6952,  0.2739, -0.9329,
         0.6161, -0.7498, -0.6177, -0.3426], grad_fn=<SelectBackward>)
--------
input word: "about"
encoding output vector: tensor([-0.7102, -0.8789, -0.8773,  0.7707, -0.9667, -0.5709, -0.8470,  0.3178,
        -0.8592,  0.1438,  0.9475,  0.8683, -0.9878, -0.2219,  0.2804,  0.3138,
        -0.8749, -0.9090,  0.0280,  0.3403,  0.3188, -0.4164, -0.6974, -0.1209,
        -0.4806, -0.7090, -0.0734,  0.2365,  0.7201, -0.9720, -0.6117,  0.0238,
        -0.1154, -0.9981,  0.3019,  0.8809,  0.7872, -0.2055,  0.9111,  0.9905,
        -0.4584,  0.8436,  0.8519, -0.9593,  0.4898, -0.1219, -0.3182, -0.8764,
        -0.9257, -0.8700, -0.3445,  0.9971,  0.7610, -0.9704, -0.9016,  0.2757,
        -0.6621,  0.8376,  0.6863, -0.9632,  0.2390, -0.8007, -0.8539,  0.7827,
        -0.3001,  0.7945,  0.1290,  0.8395, -0.7236,  0.5661,  0.5939,  0.2685,
        -0.9238,  0.6619,  0.5890, -0.2038,  0.4473, -0.9279,  0.4839, -0.8175,
         0.8447,  0.6589, -0.9363, -0.5383, -0.8423,  0.7521, -0.5931,  0.7841,
        -0.0495, -0.1247, -0.7628, -0.2167,  0.6639,  0.0163,  0.9816,  0.4988,
        -0.2257, -0.7904, -0.4764,  0.4646], grad_fn=<SelectBackward>)
--------
input word: "the"
encoding output vector: tensor([ 7.6204e-01, -7.3983e-01,  9.6650e-01, -9.6387e-01,  4.2599e-01,
         5.8212e-01,  8.7323e-01, -4.8472e-01,  7.3489e-01,  6.3324e-01,
         6.9402e-01, -9.5312e-01,  1.7038e-01, -8.6389e-01, -1.5102e-01,
        -9.6890e-01,  9.0975e-01, -9.2465e-01, -9.6505e-01,  9.7473e-01,
        -2.6594e-01, -8.8606e-01,  9.1675e-01,  7.7300e-01,  6.5380e-04,
        -9.6371e-01, -5.7857e-01, -9.0854e-01,  6.9578e-01,  9.9033e-01,
         4.4699e-01, -8.5987e-01, -5.8902e-01, -2.7272e-01,  2.5953e-01,
        -3.1288e-01, -8.8361e-01, -6.4394e-01,  4.4537e-01,  9.9756e-01,
        -9.6129e-01,  7.9536e-01, -9.9656e-01,  4.4372e-01, -8.7186e-01,
         7.4778e-01, -9.3434e-01, -5.5725e-01, -7.8864e-01, -9.1469e-01,
        -8.3310e-01,  9.9444e-01,  3.5803e-01, -6.0517e-01, -2.4733e-01,
         6.8582e-01,  7.8603e-01, -8.7760e-01,  1.0233e-02,  4.0268e-01,
         7.3582e-01,  5.8244e-01,  3.6830e-01,  2.0246e-01,  9.8354e-01,
        -5.7879e-01, -3.8296e-01, -6.1570e-01, -4.4066e-01, -5.4907e-01,
         3.6021e-01,  8.3519e-01,  6.6535e-01, -8.8805e-01,  4.7352e-01,
        -8.8248e-02, -9.8291e-01, -6.7375e-01, -9.7897e-01,  9.6412e-01,
        -9.2474e-01,  8.4459e-01, -6.6692e-01, -9.0725e-01,  8.2589e-01,
         4.0620e-01, -4.9187e-01, -3.1760e-01,  5.5942e-01, -7.9166e-01,
         8.6508e-01,  9.5651e-01, -6.4435e-01,  4.6456e-01, -1.4843e-01,
        -8.6232e-01, -2.6956e-01,  8.1615e-01, -6.9128e-01,  1.7704e-01],
       grad_fn=<SelectBackward>)
--------
input word: "Gop"
encoding output vector: tensor([ 0.4835,  0.4582,  0.2704,  0.6066,  0.3063,  0.8080, -0.0191, -0.6849,
        -0.1614, -0.1617,  0.6640,  0.2255,  0.0240, -0.8498, -0.1615, -0.9707,
         0.3624,  0.0638, -0.7192, -0.6518,  0.9040,  0.6559, -0.2976,  0.4231,
         0.7207,  0.7926, -0.6283, -0.8759,  0.9125, -0.9739, -0.8067, -0.8813,
        -0.1901, -0.5925,  0.3148,  0.1175,  0.9849, -0.8216,  0.6814,  0.0878,
         0.7214, -0.7233,  0.1435,  0.4969, -0.0817,  0.9771,  0.4377, -0.3190,
         0.4350, -0.3346,  0.1229, -0.9084, -0.2876,  0.9207, -0.9454,  0.9908,
         0.8424,  0.8910,  0.7290, -0.9259, -0.5437,  0.9708,  0.9259,  0.7330,
         0.3547, -0.6457,  0.9519, -0.8869,  0.7138,  0.4607, -0.9183,  0.8362,
        -0.9971, -0.0914, -0.7006,  0.3725, -0.0123,  0.9245, -0.9535, -0.6150,
         0.4213,  0.6954,  0.4145,  0.6751, -0.6665,  0.9006, -0.6102, -0.6044,
         0.0924,  0.8217, -0.9481, -0.2698,  0.9577, -0.6341,  0.6682,  0.7070,
         0.8351, -0.9634, -0.9218, -0.6207], grad_fn=<SelectBackward>)
--------
input word: ","
encoding output vector: tensor([ 0.1660, -0.6362, -0.8717, -0.6205, -0.0301, -0.5183,  0.7060, -0.7508,
         0.7362, -0.0687, -0.1112,  0.7012, -0.2909,  0.4415,  0.9508, -0.3729,
         0.9241, -0.0894,  0.0119, -0.4614,  0.8186,  0.8655, -0.0283, -0.6803,
         0.9625, -0.8254, -0.5455,  0.8569, -0.7644,  0.7276, -0.3906,  0.6882,
        -0.9239, -0.3192,  0.4993, -0.9816, -0.1125, -0.8155, -0.7709,  0.9598,
         0.6858,  0.9708,  0.3718,  0.7500,  0.6654, -0.9946, -0.9609,  0.8954,
         0.6858, -0.0917,  0.0712,  0.5125,  0.9298,  0.9650, -0.8161,  0.9105,
        -0.9151, -0.5974,  0.8841, -0.2437,  0.4496, -0.9784,  0.8071, -0.2166,
         0.8276,  0.9669,  0.4935,  0.7486, -0.6274,  0.2519, -0.9342, -0.9078,
        -0.3179,  0.9177,  0.5617,  0.9255, -0.4916, -0.9765,  0.8230,  0.4847,
        -0.5371, -0.1279, -0.6218, -0.7338, -0.0893, -0.8616, -0.4165, -0.1518,
         0.5755, -0.0327, -0.0977, -0.6923,  0.1690, -0.6118, -0.9606,  0.8699,
         0.3304, -0.6333,  0.7233, -0.4214], grad_fn=<SelectBackward>)
--------
input word: "they"
encoding output vector: tensor([ 0.8822,  0.1950, -0.9708,  0.9867, -0.1404,  0.4545,  0.3287, -0.8494,
         0.9369,  0.3174, -0.5030, -0.1082, -0.2595,  0.4986,  0.3017, -0.9470,
        -0.8337, -0.6647,  0.4425, -0.1921, -0.9333,  0.9887, -0.9842,  0.5715,
        -0.7528,  0.6333, -0.8734, -0.4733, -0.7520, -0.4453, -0.7698,  0.8226,
        -0.8159,  0.1460, -0.8819,  0.5136,  0.3312,  0.3552,  0.1970,  0.8844,
        -0.9713,  0.8962,  0.1355,  0.9467,  0.9553,  0.7159, -0.7193, -0.3436,
        -0.6895, -0.5260, -0.9501, -0.8891, -0.8301,  0.9236, -0.0803, -0.8788,
        -0.8907, -0.2553, -0.4004, -0.8558, -0.7051, -0.9742,  0.4764, -0.9750,
        -0.5412,  0.7216,  0.9966,  0.3639, -0.3708, -0.5300, -0.9743, -0.8453,
        -0.5785,  0.6373, -0.9938, -0.4563,  0.5603,  0.7561,  0.6345, -0.9571,
        -0.8205,  0.3534,  0.7363,  0.6152,  0.1139, -0.8615, -0.5049,  0.1978,
         0.6372,  0.1193,  0.9511,  0.9664,  0.1629, -0.5457,  0.9871, -0.1745,
        -0.7215, -0.7205,  0.9286, -0.4411], grad_fn=<SelectBackward>)
--------
input word: "boo"
encoding output vector: tensor([ 0.9671, -0.7616, -0.9386,  0.0455, -0.6501, -0.2796, -0.7583,  0.9634,
        -0.8349,  0.7102,  0.0998, -0.2912, -0.0576, -0.7418,  0.0148,  0.5413,
        -0.7260,  0.8105,  0.2969, -0.9749, -0.9201,  0.3043,  0.1540,  0.9504,
        -0.8522, -0.5210,  0.9880, -0.4670, -0.6401, -0.8089, -0.4031,  0.9980,
        -0.1109,  0.8816,  0.1187,  0.8601, -0.3043,  0.9417, -0.4995,  0.6725,
        -0.9014,  0.9082,  0.8994, -0.8529, -0.0478,  0.9898, -0.5836, -0.7569,
        -0.4734,  0.9038, -0.8977,  0.9555,  0.8013, -0.8984, -0.9241,  0.9689,
         0.9039, -0.8494, -0.2688, -0.4648,  0.3399, -0.9807, -0.6077, -0.0587,
        -0.6045,  0.9757,  0.7975,  0.9625, -0.9707,  0.4639,  0.7356,  0.9193,
        -0.3229,  0.5959, -0.4407,  0.9358,  0.9462, -0.7274,  0.3611, -0.9453,
        -0.6645, -0.2600, -0.2245,  0.9690,  0.6543, -0.0492,  0.2262,  0.1475,
        -0.8019,  0.4952,  0.1270,  0.9968, -0.3706, -0.7269, -0.2011,  0.3810,
        -0.9564, -0.9960, -0.8255,  0.6303], grad_fn=<SelectBackward>)
--------
input word: "hoo"
encoding output vector: tensor([ 0.6171,  0.8329,  0.9959, -0.9405,  0.9547,  0.1106, -0.9479, -0.3610,
        -0.8354, -0.7926, -0.6974,  0.3305, -0.9788, -0.0962, -0.9292, -0.0112,
        -0.5569,  0.2107, -0.4274,  0.0090,  0.8087,  0.6116, -0.5695, -0.9947,
        -0.6614,  0.4739,  0.1080, -0.7887,  0.0734, -0.3439,  0.2749, -0.1825,
        -0.3254,  0.8780,  0.4199, -0.8054, -0.2258,  0.3978, -0.9834, -0.0610,
        -0.3807, -0.7399,  0.7872,  0.2351,  0.5610, -0.4072,  0.8923, -0.8048,
        -0.4265, -0.9687,  0.8687, -0.9428,  0.8304, -0.5440,  0.4963, -0.6535,
        -0.2937,  0.4971,  0.9259,  0.1593,  0.0083,  0.9335,  0.8465, -0.9735,
         0.2334,  0.2327, -0.9032, -0.4137, -0.9818,  0.6093,  0.1902, -0.3185,
        -0.9525,  0.7118, -0.9207,  0.9623, -0.9598,  0.9811,  0.9016, -0.2285,
         0.6588, -0.6938, -0.7951, -0.8410,  0.8033,  0.4428, -0.7011,  0.5928,
         0.9257,  0.9558, -0.5308,  0.8060,  0.8311,  0.6866,  0.8814, -0.7781,
         0.6177,  0.6139, -0.7282, -0.7855], grad_fn=<SelectBackward>)
--------
input word: "hoo"
encoding output vector: tensor([ 0.5414,  0.9095,  0.9200, -0.5487,  0.7971,  0.8378, -0.9751, -0.8027,
        -0.9652, -0.5684, -0.9687,  0.7888, -0.4968,  0.5026, -0.7853, -0.2519,
        -0.9757,  0.4747, -0.6747,  0.7541,  0.7641,  0.2535, -0.8370, -0.9132,
         0.0477,  0.5968,  0.9329, -0.9293,  0.2821,  0.5641, -0.7738,  0.0846,
        -0.5873,  0.6909, -0.3976, -0.8852,  0.4340,  0.0720, -0.9704,  0.1128,
         0.3991, -0.8338,  0.9281, -0.2754,  0.7462, -0.9059,  0.8643, -0.7983,
        -0.6822, -0.9313,  0.9935, -0.9895,  0.7735, -0.8407, -0.4897, -0.1260,
         0.5863,  0.9589,  0.9925,  0.3926, -0.0839,  0.8259,  0.9812, -0.6665,
        -0.8353, -0.7440, -0.9812, -0.7117, -0.9752,  0.4620, -0.4713,  0.1854,
        -0.7699,  0.8580, -0.9964,  0.9027, -0.9144,  0.6863,  0.1597,  0.4901,
         0.8146, -0.9210, -0.3324, -0.9627,  0.5115,  0.2324,  0.3118, -0.3566,
         0.9664,  0.9604, -0.8953, -0.4718, -0.6167,  0.9224,  0.8087, -0.8403,
         0.0415,  0.4411, -0.9753, -0.6893], grad_fn=<SelectBackward>)
--------
input word: "and"
encoding output vector: tensor([ 0.6614, -0.9649, -0.0486,  0.9849, -0.3268,  0.6178, -0.8415, -0.9387,
        -0.6633, -0.5965, -0.9850,  0.9210,  0.7974,  0.3265,  0.2726, -0.0418,
        -0.9503,  0.2445,  0.5921, -0.8138, -0.6755,  0.9804, -0.7790,  0.7394,
         0.8457, -0.8852, -0.4023, -0.3636, -0.2346, -0.8058, -0.9720,  0.5628,
        -0.1294, -0.9213, -0.7715, -0.9819, -0.6740, -0.0801,  0.9156,  0.8466,
        -0.2242,  0.4418,  0.4831, -0.6998,  0.9983,  0.4141, -0.9343,  0.6807,
        -0.8930, -0.3911,  0.1599, -0.9736,  0.1035, -0.7889, -0.9168, -0.6179,
        -0.4853, -0.7435,  0.9746,  0.3909,  0.5489,  0.8557,  0.9739, -0.2060,
        -0.8382, -0.1417, -0.6041,  0.5288,  0.3382,  0.9690, -0.9627,  0.9868,
         0.6466,  0.9778,  0.2884, -0.3284,  0.8244,  0.1271, -0.2478,  0.6630,
         0.2488, -0.4058,  0.9578, -0.7069,  0.8172, -0.8837, -0.7598, -0.9010,
        -0.5308, -0.0317,  0.4203,  0.5901,  0.5860,  0.2227,  0.7452, -0.6556,
        -0.8342, -0.9818,  0.3670,  0.9612], grad_fn=<SelectBackward>)
--------
input word: "call"
encoding output vector: tensor([-0.0115, -0.9989,  0.4421,  0.6383, -0.9926, -0.6631,  0.8680, -0.4716,
        -0.0065, -0.4106,  0.9372,  0.6692,  0.7079,  0.8787,  0.6756, -0.9811,
         0.2234, -0.8785,  0.9282,  0.9235,  0.0113,  0.9698,  0.4994,  0.6161,
        -0.9835,  0.1148,  0.6360,  0.0016,  0.9739, -0.6967,  0.9461, -0.7856,
         0.1189, -0.8734, -0.9750, -0.0723, -0.3616, -0.7746,  0.9770,  0.6813,
        -0.8554,  0.9651,  0.4943,  0.9390,  0.8722, -0.9838, -0.0595, -0.1403,
        -0.9217, -0.9772, -0.3193,  0.9523,  0.3498, -0.2781, -0.4910,  0.4538,
        -0.3764, -0.8288,  0.8102, -0.7518, -0.3578, -0.4444,  0.3916,  0.6912,
        -0.3304,  0.1307, -0.2456, -0.2572, -0.5877,  0.9007,  0.9256, -0.7216,
        -0.3753, -0.4604,  0.8429, -0.8098, -0.9901, -0.2442, -0.0382,  0.9528,
        -0.2990,  0.6673, -0.1877, -0.1033, -0.2995, -0.7258,  0.0614,  0.1984,
         0.8924, -0.1293, -0.4539, -0.6372, -0.3761, -0.9908,  0.7452,  0.9615,
        -0.9501,  0.8432, -0.5597,  0.7606], grad_fn=<SelectBackward>)
--------
input word: "him"
encoding output vector: tensor([-0.7765, -0.7668,  0.6586,  0.0829, -0.4866,  0.0327,  0.0178, -0.8585,
         0.8371,  0.7997, -0.9902,  0.7522,  0.0380,  0.2098,  0.9801, -0.9282,
        -0.9565, -0.9912,  0.8876, -0.2727,  0.9598,  0.5703, -0.0488, -0.9536,
         0.2353,  0.9261,  0.9512, -0.7932, -0.5841,  0.1933, -0.9844,  0.0517,
        -0.8255,  0.9631,  0.9613,  0.2110, -0.6588, -0.7602,  0.7586, -0.8509,
         0.5906,  0.6515, -0.0329,  0.8256, -0.8428, -0.9002, -0.9542,  0.5539,
        -0.7687,  0.9610, -0.1400, -0.1750, -0.9072, -0.6470,  0.8488, -0.4519,
        -0.2870,  0.2269,  0.1488, -0.1438,  0.0375,  0.9370,  0.5257, -0.1307,
        -0.9114, -0.6520,  0.9671, -0.9985, -0.6903, -0.9560, -0.9444, -0.9879,
         0.0020,  0.0057, -0.0446,  0.9701, -0.9241, -0.9837,  0.8127,  0.5768,
        -0.2577, -0.4596,  0.6681, -0.8971,  0.2220, -0.4302,  0.7102,  0.1097,
        -0.5676, -0.9904, -0.1307,  0.6812, -0.0621, -0.8475, -0.1178, -0.8804,
         0.9765,  0.9745,  0.9698,  0.9356], grad_fn=<SelectBackward>)
--------
input word: "a"
encoding output vector: tensor([ 0.8784, -0.5792, -0.9823,  0.9912, -0.8285, -0.9871, -0.9920,  0.4061,
         0.3207, -0.6849, -0.7414,  0.7330,  0.7833, -0.5985,  0.8168, -0.9210,
        -0.9730,  0.5504, -0.7729, -0.9772, -0.8198,  0.8784,  0.9877,  0.6055,
         0.2988,  0.8486, -0.6578,  0.2618, -0.7941,  0.0074, -0.8795,  0.3837,
         0.5790,  0.9788, -0.9635, -0.2951,  0.7554, -0.3035,  0.4263, -0.4856,
        -0.9996,  0.3823,  0.3216,  0.9399,  0.7609,  0.4285,  0.1332, -0.7626,
        -0.9916, -0.6487, -0.5844, -0.6152, -0.9756, -0.9791, -0.4364,  0.6024,
        -0.2212, -0.2462,  0.8103, -0.9834, -0.6680, -0.5449, -0.8740, -0.5634,
         0.1813,  0.8496,  0.0779,  0.5988,  0.2056, -0.9620,  0.7283,  0.7695,
         0.8674,  0.5166, -0.1673,  0.9669,  0.4512, -0.7964,  0.9436, -0.8662,
        -0.3365,  0.5813, -0.7450,  0.9144, -0.8206,  0.5471,  0.0347,  0.9609,
         0.9930, -0.7702, -0.9418,  0.9786, -0.2203, -0.7792, -0.2106, -0.8899,
        -0.9413, -0.7894,  0.9810,  0.9657], grad_fn=<SelectBackward>)
--------
input word: "bully"
encoding output vector: tensor([ 0.2384, -0.8897, -0.3554,  0.8390, -0.4614, -0.8213, -0.0326,  0.9584,
        -0.7714, -0.9939,  0.7638, -0.0369, -0.9347, -0.9303, -0.7595,  0.7817,
         0.9472,  0.9379, -0.7506, -0.9904,  0.4198, -0.6654, -0.2711, -0.9695,
        -0.6155,  0.0246,  0.1665, -0.6949, -0.1713, -0.9739, -0.2769,  0.9850,
         0.3193, -0.0747,  0.8924,  0.9811,  0.8862,  0.9920, -0.1859,  0.3729,
         0.4787,  0.0597,  0.8119, -0.9157, -0.8531,  0.8362,  0.4519,  0.4683,
        -0.6696,  0.6438, -0.7899, -0.8526, -0.0649, -0.2385, -0.0592, -0.3249,
        -0.9404, -0.9069, -0.7380,  0.9174,  0.4002,  0.2225, -0.7452, -0.8031,
         0.4038, -0.9068, -0.7032,  0.5416, -0.9850, -0.2717,  0.6568, -0.0471,
        -0.2049,  0.7367, -0.6410, -0.6436,  0.8066,  0.2917,  0.9044, -0.8289,
         0.8604, -0.5746, -0.5249,  0.8335,  0.6964, -0.5829, -0.4253,  0.7369,
         0.7781,  0.4088, -0.2460, -0.2921, -0.9033, -0.0566,  0.9732,  0.8616,
        -0.7170,  0.1659, -0.0344,  0.1472], grad_fn=<SelectBackward>)
--------
input word: "."
encoding output vector: tensor([-0.8967,  0.8047,  0.8634, -0.9624,  0.8917, -0.7203,  0.9772,  0.9280,
         0.6539, -0.5019, -0.4340, -0.4231, -0.8180, -0.7600, -0.7909,  0.9629,
        -0.8893, -0.9399,  0.6796, -0.3210, -0.3763, -0.9933,  0.9770, -0.7728,
         0.9390,  0.8150, -0.9901,  0.1256,  0.7016,  0.4432, -0.9311,  0.1057,
         0.0328,  0.3518, -0.3876,  0.8604, -0.8582, -0.8455, -0.5599,  0.5177,
         0.2653, -0.5135, -0.7347, -0.0601, -0.9714,  0.9712,  0.1876,  0.3801,
         0.2998, -0.7476, -0.6833,  0.3204,  0.9109, -0.9610,  0.8430,  0.7855,
        -0.9135, -0.9092,  0.9973,  0.9219,  0.2251,  0.0789, -0.3841, -0.9580,
         0.9602, -0.4025,  0.9686, -0.2419, -0.8856,  0.0351,  0.8664, -0.8042,
         0.4589,  0.9194,  0.3501,  0.4705, -0.4325,  0.8083, -0.6933, -0.9354,
         0.5133, -0.4956,  0.6396,  0.9359,  0.7660,  0.5032,  0.5072, -0.5048,
        -0.8058, -0.1387,  0.9879,  0.9606,  0.8299, -0.7447,  0.9323,  0.0823,
         0.0367,  0.1534, -0.2728,  0.1725], grad_fn=<SelectBackward>)
--------
```


  - The Encoder final state hidden vector which is the single vector for sequence:

```
single vector tensor([[[-0.8967,  0.8047,  0.8634, -0.9624,  0.8917, -0.7203,  0.9772,
           0.9280,  0.6539, -0.5019, -0.4340, -0.4231, -0.8180, -0.7600,
          -0.7909,  0.9629, -0.8893, -0.9399,  0.6796, -0.3210, -0.3763,
          -0.9933,  0.9770, -0.7728,  0.9390,  0.8150, -0.9901,  0.1256,
           0.7016,  0.4432, -0.9311,  0.1057,  0.0328,  0.3518, -0.3876,
           0.8604, -0.8582, -0.8455, -0.5599,  0.5177,  0.2653, -0.5135,
          -0.7347, -0.0601, -0.9714,  0.9712,  0.1876,  0.3801,  0.2998,
          -0.7476, -0.6833,  0.3204,  0.9109, -0.9610,  0.8430,  0.7855,
          -0.9135, -0.9092,  0.9973,  0.9219,  0.2251,  0.0789, -0.3841,
          -0.9580,  0.9602, -0.4025,  0.9686, -0.2419, -0.8856,  0.0351,
           0.8664, -0.8042,  0.4589,  0.9194,  0.3501,  0.4705, -0.4325,
           0.8083, -0.6933, -0.9354,  0.5133, -0.4956,  0.6396,  0.9359,
           0.7660,  0.5032,  0.5072, -0.5048, -0.8058, -0.1387,  0.9879,
           0.9606,  0.8299, -0.7447,  0.9323,  0.0823,  0.0367,  0.1534,
          -0.2728,  0.1725]]], grad_fn=<UnsqueezeBackward0>)
single vector shape: torch.Size([1, 1, 100])
-----------------------------------------------
```

  - The above single vector is passed as an input to the first LSTM node.
  - The output of this first LSTM node is passed as input to the next LSTM node.
  - The hidden state and cell state is initialised to zero and is passed to the next LSTM layer.


```
Decoder output after 1st step
decoder_output tensor([[[ 0.1527, -0.2696,  0.0396, -0.2578,  0.0822, -0.1594,  0.4237,
           0.0545, -0.2702,  0.1662, -0.2275, -0.1185,  0.0542,  0.0135,
           0.1288,  0.0529,  0.0903,  0.0033,  0.3039,  0.0081, -0.3846,
           0.1927, -0.0654, -0.0114, -0.2742, -0.0431,  0.3285,  0.1084,
           0.2223, -0.0455, -0.1328, -0.0201,  0.1317, -0.0191, -0.0407,
           0.1213, -0.0146,  0.0368,  0.0521, -0.1465, -0.1475, -0.0400,
          -0.2144,  0.3432, -0.0691, -0.0922,  0.0301,  0.1245,  0.0072,
           0.0234, -0.0208, -0.1518, -0.0789, -0.1248,  0.4189,  0.0337,
           0.0817,  0.0317,  0.0220,  0.1389, -0.0789,  0.0666, -0.0293,
          -0.2056, -0.1911,  0.2015, -0.2406, -0.1974,  0.0709,  0.1053,
          -0.0355, -0.2285, -0.0432, -0.2492, -0.2627, -0.1261, -0.0555,
           0.0861,  0.1658,  0.3124, -0.0446, -0.1377, -0.0531,  0.2247,
          -0.1028,  0.1529,  0.0709, -0.0578, -0.0114, -0.2276,  0.1861,
          -0.0470, -0.0474,  0.1546,  0.3176, -0.2259, -0.1188, -0.3154,
           0.3268,  0.0360]]], grad_fn=<TransposeBackward0>)

decoder_output shape: torch.Size([1, 1, 100])
-----------------------------------------------

Decoder output after 2nd step

Decoder output after 2nd step
decoder_output tensor([[[ 0.1509, -0.1406, -0.0447, -0.1335,  0.1882, -0.0886,  0.0274,
           0.1447, -0.1553,  0.1853, -0.4539, -0.0218,  0.0077, -0.0134,
           0.0362,  0.0819, -0.0046, -0.0158,  0.1126, -0.0438, -0.1247,
           0.0317, -0.0325, -0.0513, -0.2032, -0.0127,  0.1752,  0.0322,
           0.2894, -0.1126, -0.0341, -0.0144,  0.1431, -0.0506,  0.2789,
           0.0182, -0.1197,  0.1100,  0.1045,  0.0628, -0.1174,  0.0520,
          -0.1170,  0.1332, -0.1188, -0.1095, -0.0137,  0.0692,  0.0787,
          -0.0208,  0.0991, -0.0149,  0.0522, -0.0964,  0.1504,  0.1116,
           0.0622,  0.0683,  0.0602,  0.0921, -0.0534,  0.0065,  0.0098,
           0.0178, -0.1351,  0.1162, -0.1293, -0.0249, -0.0381,  0.0314,
           0.0842, -0.0422, -0.0143, -0.0669, -0.1494, -0.0700,  0.1115,
           0.1280,  0.0783,  0.1350,  0.1255, -0.0824, -0.1109,  0.0501,
          -0.1284,  0.0411,  0.0866, -0.0655, -0.0717, -0.0520,  0.0246,
           0.0816,  0.0348,  0.0672,  0.0885, -0.0375,  0.0182, -0.1220,
           0.1560,  0.0144]]], grad_fn=<TransposeBackward0>)

decoder_output2 shape: torch.Size([1, 1, 100])
```
