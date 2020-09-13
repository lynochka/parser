import json
import os
from parser.create_dataset import NumbersDataset
from parser.numbers_model import NumbersModel
from pathlib import Path

import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE

max_sequence_length = 10


def main():
    # DATALOAD: LOAD TEXT, TARGETS, TOKENIZER, ENCODER
    text_train_dataset = tf.data.experimental.CsvDataset(
        filenames=os.path.join("data", "training_data.csv"),
        record_defaults=[tf.string, tf.float32],
        header=False,
    )
    text_validation_dataset = tf.data.experimental.CsvDataset(
        filenames=os.path.join("data", "validation_data.csv"),
        record_defaults=[tf.string, tf.float32],
        header=False,
    )

    tokenizer, encoder = NumbersDataset.load_tokenizer_and_encoder_from_metadata(
        os.path.join("data", "tokenizer_metadata.json"),
        os.path.join("data", "encoder_metadata.json"),
    )

    # DATAPREP: ENCODE SEQUENCES FROM TEXT
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

    batch_size = 10
    validation_batch_size = 100
    buffer_size = 100

    # add prefetch for large dataset
    train_dataset = (
        text_train_dataset.map(encode_pyfn, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size, seed=0, reshuffle_each_iteration=True)
        .padded_batch(batch_size=batch_size, padded_shapes=(max_sequence_length, []))
    )

    validation_dataset = (
        text_validation_dataset.map(encode_pyfn, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size, seed=1, reshuffle_each_iteration=True)
        .padded_batch(
            batch_size=validation_batch_size, padded_shapes=(max_sequence_length, [])
        )
    )

    # MODEL: TRAIN
    model = NumbersModel(encoder.vocab_size, max_sequence_length)

    checkpoints_path = os.path.join(".", "checkpoints")
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    best_model_filepath = os.path.join(checkpoints_path, "best_model.hdf5")
    best_model_metadata = os.path.join(checkpoints_path, "best_model.json")

    # NOTE: article implementation stops after 300K gradient descent steps
    epochs = int(1e5)
    num_gradient_steps = int(3e5)

    gradient_step = 0
    patience = 0  # track for now
    best_test_loss = 1e8
    history = []

    for epoch in range(epochs):
        if gradient_step > num_gradient_steps:
            break
        # Reset the metrics at the start of the next epoch
        model.reset_states()

        for encoded_sequences, targets in train_dataset:
            model.train_step(encoded_sequences, targets)
            gradient_step += 1

        for v_encoded_sequences, v_targets in validation_dataset:
            model.test_step(v_encoded_sequences, v_targets)

        loss = model.train_loss.result()
        test_loss = model.test_loss.result()
        if test_loss < best_test_loss:
            patience = 0
            best_test_loss = test_loss
            model.save_weights(best_model_filepath, overwrite=True)
            with open(best_model_metadata, "w") as f:
                json.dump({"epoch": epoch, "history": history}, f)
        else:
            patience += 1

        history.append(
            {
                "loss": float(loss.numpy()),
                "test_loss": float(test_loss.numpy()),
                "mae": float(model.train_mae.result().numpy()),
                "test_mae": float(model.test_mae.result().numpy()),
                "patience": patience,
            }
        )

        if epoch % 100 == 0:
            template = (
                "Epoch {}, Steps {}, "
                + "MAE: {}, Test MAE: {}, "
                + "Loss: {}, Test Loss: {}, "
                + "Patience: {}, Best Test Loss: {}"
            )
            print(
                template.format(
                    epoch,
                    gradient_step,
                    model.train_mae.result(),
                    model.test_mae.result(),
                    loss,
                    test_loss,
                    patience,
                    best_test_loss,
                )
            )


if __name__ == "__main__":
    main()
