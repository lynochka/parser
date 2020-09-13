import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Model


class NALU(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super(NALU, self).__init__()
        self.units = units
        self.W_hat = None
        self.M_hat = None
        self.G = None
        self.matrix_initializer = tf.keras.initializers.GlorotNormal()
        self.epsilon = 1e-7

    def build(self, input_shape):
        input_dim = input_shape[-1]
        shape = (input_dim, self.units)
        self.W_hat = self.add_weight(
            "W_hat", shape=shape, initializer=self.matrix_initializer, trainable=True
        )
        self.M_hat = self.add_weight(
            "M_hat", shape=shape, initializer=self.matrix_initializer, trainable=True
        )
        self.G = self.add_weight(
            "G", shape=shape, initializer=self.matrix_initializer, trainable=True
        )
        self.built = True

    def call(self, inputs, **kwargs):
        # NAC
        W = tf.tanh(self.W_hat) * tf.sigmoid(self.M_hat)
        a = tf.matmul(inputs, W)
        # NALU
        g = tf.sigmoid(tf.matmul(inputs, self.G))
        m = tf.math.exp(tf.matmul(tf.math.log(tf.math.abs(inputs) + self.epsilon), W))
        return g * a + (1 - g) * m


class NumbersModel(Model):
    def __init__(self, vocabulary_size=30, max_sequence_length=10):
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
        self.nalu = NALU(1)

        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.loss_object = tf.keras.losses.MeanAbsoluteError()

        self.train_metric = tf.keras.metrics.Mean(name="train_metric")
        self.test_metric = tf.keras.metrics.Mean(name="test_metric")
        self.metric_object = tf.keras.losses.MeanSquaredError()

    def reset_states(self):
        self.train_loss.reset_states()
        self.test_loss.reset_states()
        self.train_metric.reset_states()
        self.test_metric.reset_states()

    def call(self, x, **kwargs):
        x = self.embedding(x)
        x = self.lstm(x)
        return self.nalu(x)

    @tf.function
    def train_step(self, encoded_sequences, targets):
        with tf.GradientTape() as tape:
            predictions = self(encoded_sequences)
            loss = self.loss_object(targets, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        self.train_metric(self.metric_object(targets, predictions))

    @tf.function
    def test_step(self, encoded_sequences, targets):
        predictions = self(encoded_sequences)
        t_loss = self.loss_object(targets, predictions)
        self.test_loss(t_loss)
        self.test_metric(self.metric_object(targets, predictions))
