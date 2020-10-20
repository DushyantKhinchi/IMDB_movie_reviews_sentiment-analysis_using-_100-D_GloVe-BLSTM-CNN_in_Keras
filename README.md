# IMDB_movie_reviews_sentiment-analysis_using-_100-D_GloVe-BLSTM-CNN_in_Keras

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/1_AaSAGUqdtt0SFfz2i9Hr3w.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)


Dataset used in this project has 50,000 IMDB reviews with binary sentiment labels represented by '1' and '0' where 1 being the 'Positive sentiment' and 0 being the 'negative sentiment'. The data set was derived from the ACL 2011 paper. 

---
#Goals

* Preprocessing of the text data.
* Constructing the word embedding matrix.
* Dimensionality reduction and Visualization of word vectors using the t-SNE algorithm
* Constructing Bidirectional LSTM architecture
* Constructing CNN architecture
* Tuning the models
* Visualizing accuracy and loss plots
---

First, all the reviews in the 'text' column were tokenized using Keras 'Tokenizer' function and the internal vocabulary was updated using a list of texts. Later, the embedding matrix of shape 124253 X 100 was constructed using 100-dimensional GloVe word embeddings and word vectors were visualized using t-SNE algorithm which basically reduces the dimensionality of a higher dimensional vector into 2 or 3 dimensions thus making the visualization possible. Here, we've reduced the vector dimensions from 100 to 2.

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/word_vectors%20(1).png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)

## Bi-LSTM
Embedding layer was constructed and a Bidirectional LSTM model was defined with 30 LSTM cells, Global Max Pooling 1D layer, a dense layer with 30 units, and a dropout layer.

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/Screenshot_4-0001.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)

Some hyperparameter values used to compile the model consists of 'sparse_categorical_crossentropy' as loss and  'Adam' optimizer with a learning rate of 0.01

The model was trained for 10 epochs with a batch size of 512.

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/LSTM_loss.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/LSTM_accuracy.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)


Final metric values for BLSTM model

* Train loss: 0.0472
* Train accuracy: 98.49% 
* Validation loss: 0.4018
* Validation accuracy: 89.43% 

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/Screenshot_5-0001.png/:/rs=w:1280)

## CNN
Now, CNN model was constructed with Conv1D layer having 32 filters, followed by maxpooling1D, and an LSTM layer with 100 cells.

(https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/Screenshot_6-0001.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)

Some hyperparameter values used to compile the model consists of 'binary_crossentropy' as loss and  'Adam' optimizer with a learning rate of 0.01

(https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/CNN_loss.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)

(https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/CNN_accuracy.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)

Final metric values for CNN model

* Train loss: 0.0536
* Train accuracy: 98.37%
* Validation loss: 0.3996
* Validation accuracy: 89.58% 

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/cnn-0001.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)

![alt text](https://img1.wsimg.com/isteam/ip/6accb018-0248-4224-8307-a1bd01733c4f/Screenshot_7-0001.png/:/cr=t:0%25,l:0%25,w:100%25,h:100%25/rs=w:1280)

---

 Key Outcomes 

* Dataset of 50000 IMDB movie reviews was used as 70:30 test-train split to develop neural network models.
* 100-dimensional GloVe dictionary was used to create word embeddings.
* TSNE algorithm was used to visualize word embeddings in a 2-dimensional plane.
* Bidirectional LSTM model was developed that resulted in train loss of 0.0472, train accuracy of 98.49%, and a test accuracy of 89.43%
* CNN model was developed that resulted in a train loss of 0.0536, train accuracy of 98.37%, and a test accuracy of 89.58% 
