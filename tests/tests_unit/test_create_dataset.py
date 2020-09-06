from parser.create_dataset import NumbersDataset


def test_dataset_correct_init_split():
    numbers_dataset = NumbersDataset(seed=0)

    training_numbers = set(number for _, number in numbers_dataset.training_data)
    validation_numbers = set(number for _, number in numbers_dataset.validation_data)
    test_numbers = set(number for _, number in numbers_dataset.test_data)

    # no intersection between the sample numbers
    assert len(training_numbers.intersection(validation_numbers)) == 0
    assert len(training_numbers.intersection(test_numbers)) == 0
    assert len(validation_numbers.intersection(test_numbers)) == 0

    assert set(range(1, 1000)) == training_numbers.union(validation_numbers).union(
        test_numbers
    )
