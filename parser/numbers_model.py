import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Model


class NumbersModel(Model):
    def __init__(self, vocabulary_size, max_sequence_length):
        super(NumbersModel, self).__init__()

        embedding_dim = 2
        lstm_dim = 3

        self.embedding = Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            mask_zero=True,
            input_length=max_sequence_length,
            trainable=True,
        )
        self.lstm = LSTM(lstm_dim, dropout=0, recurrent_dropout=0)
        self.d = Dense(1)

        nalu_initializer = tf.keras.initializers.GlorotNormal()
        nalu_shape = (lstm_dim, 1)
        self.W_hat = tf.Variable(nalu_initializer(shape=nalu_shape))
        self.M_hat = tf.Variable(nalu_initializer(shape=nalu_shape))
        self.G = tf.Variable(nalu_initializer(shape=nalu_shape))

    def _nalu(self, x):
        epsilon = 1e-7

        # NAC
        W = tf.tanh(self.W_hat) * tf.sigmoid(self.M_hat)
        a = tf.matmul(x, W)
        # NALU
        g = tf.sigmoid(tf.matmul(x, self.G))
        m = tf.math.exp(tf.matmul(tf.math.log(tf.math.abs(x) + epsilon), W))
        return g * a + (1 - g) * m

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        return self._nalu(x)
