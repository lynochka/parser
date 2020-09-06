# Work in progress

Implementation of **Language to Number Translation Tasks** based on NALU [[source](https://arxiv.org/pdf/1808.00508/)].

## Environment

To istall poetry follow https://python-poetry.org/docs/#installation

To intialize the environment
```sh
poetry config virtualenvs.in-project true
poetry install
```

## Data preparation
To reproduce the article data

```python
from parser.create_dataset import NumbersDataset

numbers_dataset = NumbersDataset()
numbers_dataset.dump_data("data")
```

## TODO: Model
