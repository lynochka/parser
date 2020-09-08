# Language to Number Translation

Implementation of **Language to Number Translation Tasks** based on NALU [[source](https://arxiv.org/pdf/1808.00508/)].

## Environment

To istall poetry follow https://python-poetry.org/docs/#installation

To intialize the environment
```sh
poetry config virtualenvs.in-project true
poetry install
```

## Create data
To reproduce the article data

```python
from parser.create_dataset import NumbersDataset

numbers_dataset = NumbersDataset()
numbers_dataset.dump_data("data")
```

## Create model (work in progress)

* `numbers_model.py` includes the model class
* `create_model.py` includes the data preparation and model training (to be completed with e.g., model saving, experiment tracking)
