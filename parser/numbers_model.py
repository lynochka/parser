import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine import compile_utils


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

        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.train_mae = tf.keras.metrics.Mean(name="train_mae")
        self.test_mae = tf.keras.metrics.Mean(name="test_mae")
        self.mae_object = tf.keras.losses.MeanAbsoluteError()

    def reset_states(self):
        self.train_loss.reset_states()
        self.test_loss.reset_states()
        self.train_mae.reset_states()
        self.test_mae.reset_states()

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

    @tf.function
    def train_step(self, encoded_sequences, targets):
        with tf.GradientTape() as tape:
            predictions = self(encoded_sequences)
            loss = self.loss_object(targets, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        self.train_mae(self.mae_object(targets, predictions))

    @tf.function
    def test_step(self, encoded_sequences, targets):
        predictions = self(encoded_sequences)
        t_loss = self.loss_object(targets, predictions)
        self.test_loss(t_loss)
        self.test_mae(self.mae_object(targets, predictions))
