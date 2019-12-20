======================
A prototype + an experimental Speech-to-Text Model 
#### (modified ver. of A. Pal)
======================
Implementation
Dataset
I use the LibriSpeech corpus of audiobook data to train and evaluate models. I use the train-clean-100 and train-clean-360 subsets (~460 hours, >1000 speakers) for training and the dev-clean subset (~10 hours, 40 speakers for validation. This data contains audio from controlled environments no external noise just recording artifacts such as microphone buzz. The LibriSpeech corpus is available free of charge.
In all experiments I downsample the audio from 16 KHz to 4 KHz for quicker experimentation and reduced computational load.
Model
I use a simple 4 layer convolutional encoder network throughout this work. Architecture is as follows
1D convolution with large (size 32) filters followed by batch normalisation and max pooling with size 4 and stride 4
3 times 1D convolutions of size 3 followed batch normalisation and max pooling with size 2 and stride 2
Global max pooling and a dense layer to produce the embeddings
The siamese network is two of the above networks (with weight sharing) joined by a euclidean distance layer
The final layer is a dense layer with sigmoid activation
I use ReLu activation everywhere except the final layer. I choose to perform quite aggressive max-pooling in the first layer as the initial size of the samples is quite large.

Training
In each experiment I trained a siamese networks on batches of 32 similar and 32 dissimilar pairs for 50 “epochs” (each epoch is 1000 batches). The per-pair labels are 0 for similar pairs and 1 for dissimilar pairs i.e. pairs from same and different speakers.
I used the Adam optimizer with a learning rate of 0.001 of throughout. The 1-shot classification accuracy on a held out validation set of previously unseen speakers was used to determine the learning rate schedule — when this metric plateaus for 10 epochs the learning rate is dropped by a factor of 10.
Audio fragment length tuning
One key parameter is the length of the audio to feed as input to the model. Intuitively one expects the accuracy of the model to increase as it is fed richer input data however the trade-off is increased training time and memory requirements.

Audio fragment length vs validation-set metrics
One can see that both validation 1-shot accuracy and verification accuracy keep on increasing with audio fragment length. I selected a length of 3 seconds as there appears to be diminishing returns after this point.
Hyperparameter grid search
I performed a grid search on the following hyperparameter space: initial number of convolutional filters (16, 32, 64, 128), embedding dimension (32, 64, 128, 256, 512) and dropout fraction (0, 0.1). Best results were achieved with 128 filters, embedding dimension of 64 and no dropout.
Results
Below are the results of the best siamese network on 1-shot, k-way and 5-shot, k-way classification tasks for 2 ≤ k ≤ 20. I also trained a classifier network on the same dataset, using the same architecture and hyperparameters as a single encoder “twin” with the addition of a 1172 way softmax after the bottleneck layer and a categorical crossentropy loss.

There are two results to note:
The siamese network does (slightly) outperform the classifier bottleneck embedding at 1-shot classification as hoped
Unexpectedly the classifier bottleneck embedding performs better at 5-shot classification than the siamese network with identical hyperparams
Note that the 5-shot classification performance gap between the classifier bottleneck embedding and the siamese network embedding is larger for higher k. Potentially this could be because classifier learns a better embedding for distinguishing between many classes due to the nature of the labels it uses when training. Consider that when using one-hot classification labels the softmax score of the correct speaker is pushed up and the softmax scores of all the other speakers is pushed down.
Compare this to verification tasks where training pushes the embedding distance between samples of different speakers apart, but only for the different speakers present in the batch. In particular I expect the siamese network to quickly learn to differentiate between male and female voices and that feeding it opposite-gender pairs will provide almost no learning in later epochs because they will be “too easy”. One approach to combat this would be to perform hard-negative mining to create more difficult verification tasks.
Embedding space visualisation

Circle = male, triangle = female. Each colour corresponds to a different speaker identity.
Shown above is a visualisation of the embedding space learnt by the siamese network. I selected 20 speakers from the training set and random and for each of these I selected 10 random audio samples. I then used tSNE to embed the 64-dimensional points that represent audio samples into a 2-dimensional space for plotting. Note that there is good clustering of speaker identities (as represented with colour) and a general separation of male and female speakers (as indicated by circular or triangular markers).
