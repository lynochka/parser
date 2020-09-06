from typing import Dict, List, Set, Tuple

import inflect
import numpy as np
import tensorflow_datasets as tfds

p = inflect.engine()


class NumbersDataset:
    def __init__(self, seed=0):
        """
        We follow the NALU article for data generation
        """

        self.np_random_seed = np.random.RandomState(seed)

        self.tokenizer: tfds.core.features.text.text_encoder.Tokenizer = (
            tfds.features.text.Tokenizer()
        )

        # NOTE: NALU article uses validation data to choose a better model
        self.training_data = []  # type: List[Tuple[string, float]]
        self.validation_data = []  # type:List[Tuple[string, float]]
        self.test_data = []  # type: List[Tuple[string, float]]

        number_to_text: Dict[int, str] = dict()
        number_to_token_set: Dict[int, Set[str]] = dict()

        # mapping of tokens to ensure all unique tokens are added to training
        missing_training_tokens: Set[str] = set()
        training_token_set: Set[str] = set()

        for number in range(1, 1000):
            number_text = p.number_to_words(number)
            number_to_text[number] = number_text

            token_set = set(self.tokenizer.tokenize(number_text))
            number_to_token_set[number] = set(token_set)

            # add all numbers < 20 to training data, remove them from missing
            if number < 20:
                self.training_data.append((number_to_text[number], number))
                training_token_set.update(token_set)
                continue
            # add missing training tokens to be sampled into training
            for token in token_set:
                if token not in training_token_set:
                    missing_training_tokens.add(token)

        # following NALU paper take 630 numbers into test, and 200 into validation
        missing_numbers = self.np_random_seed.permutation(np.arange(20, 1000))

        for number in missing_numbers:
            text: str = number_to_text[number]
            tokens: Set[str] = number_to_token_set[number]
            if tokens.difference(training_token_set):
                self.training_data.append((text, number))
                training_token_set.update(tokens)
                continue
            if len(self.test_data) < 630:
                self.test_data.append((text, number))
                continue
            if len(self.validation_data) < 200:
                self.validation_data.append((text, number))
                continue

            self.training_data.append((text, number))
