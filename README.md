# Language to Number Translation

Implementation of **Language to Number Translation Tasks** based on NALU [[source](https://arxiv.org/pdf/1808.00508/)].

## Environment

To install poetry follow https://python-poetry.org/docs/#installation

To initialize the environment
```sh
poetry config virtualenvs.in-project true
poetry install
```

## Data
To reproduce the article data, and recreate data encoder and tokenizer.

```sh
poetry run python parser/create_dataset.py
```

## Model

`parser/numbers_model.py` includes the model class and the custom NALU layer.

To train the model, while keeping track of the best model.
```sh
poetry run python parser/train_model.py
```

## Prediction (in progress)

`parser/simple_predict.py` includes an example of prediction per single text input.
