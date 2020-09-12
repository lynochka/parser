from parser.create_dataset import NumbersDataset
from parser.numbers_model import NumbersModel

import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE

# CREATE & SAVE DATA
numbers_dataset = NumbersDataset()
numbers_dataset.dump_data("data")

# TEXT TO ENCODED SEQUENCES
# TODO: c>hange how encoder is created and stored while dataset could be added to
encoder = numbers_dataset.encoder
max_sequence_length = 10


def encode(text, target, max_len=max_sequence_length):
    text_encoded = encoder.encode(text.numpy())[:max_len]
    return text_encoded, target


def encode_pyfn(text, target):
    text_encoded, target = tf.py_function(
        encode, inp=[text, target], Tout=(tf.int32, tf.float32)
    )
    text_encoded.set_shape([None])
    target.set_shape([])
    return text_encoded, target


text_train_dataset = tf.data.experimental.CsvDataset(
    filenames="data/training_data.csv",
    record_defaults=[tf.string, tf.float32],
    header=False,
)
text_validation_dataset = tf.data.experimental.CsvDataset(
    filenames="data/validation_data.csv",
    record_defaults=[tf.string, tf.float32],
    header=False,
)

batch_size = 10
buffer_size = 1000

# add prefetch for large dataset
train_dataset = (
    text_train_dataset.map(encode_pyfn, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size, seed=0, reshuffle_each_iteration=True)
    .padded_batch(batch_size=batch_size, padded_shapes=(max_sequence_length, []))
)

validation_dataset = (
    text_validation_dataset.map(encode_pyfn, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size, seed=1, reshuffle_each_iteration=True)
    .padded_batch(batch_size=batch_size, padded_shapes=(max_sequence_length, []))
)

model = NumbersModel(numbers_dataset.encoder.vocab_size, max_sequence_length)

# NOTE: article implementation stops after 300K gradient descent steps
EPOCHS = int(1e4)
num_gradient_steps = int(3e5)
gradient_step = 0

for epoch in range(EPOCHS):
    if gradient_step > num_gradient_steps:
        break
    # Reset the metrics at the start of the next epoch
    model.reset_states()

    for encoded_sequences, targets in train_dataset:
        model.train_step(encoded_sequences, targets)

    for v_encoded_sequences, v_targets in validation_dataset:
        model.test_step(v_encoded_sequences, v_targets)

    if not ((epoch + 1) % 10 == 0):
        continue
    template = "Epoch {}, Steps {}, Loss: {},  Test Loss: {}"
    print(
        template.format(
            epoch + 1,
            gradient_step,
            model.train_loss.result(),
            model.test_loss.result(),
        )
    )
