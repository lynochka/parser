import os
from parser.create_dataset import NumbersDataset
from parser.numbers_model import NumbersModel
from parser.train_model import max_sequence_length

import tensorflow as tf


def simple_encode(text, max_len=max_sequence_length):
    sequence = encoder.encode(text)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
        [sequence],
        maxlen=max_len,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0,
    )
    return sequences


def simple_predict(model, text):
    sequences = simple_encode(text)
    return model(sequences).numpy()[0][0]


tokenizer, encoder = NumbersDataset.load_tokenizer_and_encoder_from_metadata(
    os.path.join("data", "tokenizer_metadata.json"),
    os.path.join("data", "encoder_metadata.json"),
)

model = NumbersModel(encoder.vocab_size, max_sequence_length)
# HACK: intialize the model
simple_predict(model, "one")

model.load_weights(os.path.join("checkpoints", "best_model.hdf5"))

text = "hundred and eighteen"
print(simple_predict(model, text))
