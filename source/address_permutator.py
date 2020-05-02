import numpy as np
from tqdm import tqdm
from numpy.testing import assert_almost_equal
from random import shuffle, choice
from functools import reduce


class AddressPermutator:
    def __init__(self, addresses):
        self.addresses = addresses
        self.encoding = None
        self.decoding = None

        self._validate_addresses()

    def permutate(self, permutation_buckets=(0.6, 0.3, 0.1)):
        self._validate_permutation_buckets(permutation_buckets)
        random_seed = np.random.uniform(size=(len(self.addresses)))
        permutated_addresses = []
        for i, row in tqdm(enumerate(self.addresses)):
            if random_seed[i] <= permutation_buckets[0]:
                permutated_addresses.append(self._concat_row(row))
            elif random_seed[i] <= sum(permutation_buckets[:2]):
                permutated_addresses.append(self._mild_permutation(row))
            elif random_seed[i] <= sum(permutation_buckets):
                permutated_addresses.append(self._erratic_permutation(row))
            else:
                raise
        standard_addresses = tuple(map(self._concat_row, self.addresses))

        return tuple(permutated_addresses), standard_addresses

    def encode(self, permutated, standard):
        all_text = permutated + standard
        maxlen = len(max(all_text, key=len))

        try:
            assert len(permutated) == len(standard)
        except AssertionError:
            raise AssertionError('Iterables must have same length.')

        big_string = "".join(all_text)
        all_letters = tuple(set(big_string))

        self.encoding = {key: value for value, key in enumerate(all_letters)}
        self.decoding = {value: key for value, key in enumerate(all_letters)}

        permutated_long = [x.ljust(maxlen) for x in permutated]
        standard_long = [x.ljust(maxlen) for x in standard]
        X = self._assign_bools(permutated_long, maxlen, self.encoding)
        y = self._assign_bools(standard_long, maxlen, self.encoding)

        return X, y

    def decode(self, encoded_matrix):
        decoded = []
        for row in encoded_matrix:
            decode_map = map(lambda x: self.decoding[np.argmax(x)], row)
            decoded.append("".join(decode_map))

        return decoded

    def _validate_addresses(self):
        try:
            assert type(self.addresses) == np.ndarray
        except AssertionError as e:
            raise e('Addresses must be passed as matrix.')

    def _validate_permutation_buckets(self, buckets):
        for bucket in buckets:
            try:
                assert (type(bucket) == float) or (bucket == 0)
            except AssertionError:
                raise AssertionError(
                    'All buckets must be of type float or be equal to 0.'
                    )

        try:
            assert_almost_equal(sum(buckets), 1)
        except AssertionError:
            raise AssertionError('Buckets sum must be equal to 1.')

    def _erratic_permutation(self, row):
        delimiters = (',', '.', ' ', '|', ':', ';')
        shuffled_row = row.copy()
        shuffle(shuffled_row)
        shuffled_address = self._concat_row(shuffled_row)
        result = self._permutate_delimiters(
            shuffled_address,
            delimiters
            )
        return result

    def _mild_permutation(self, row):
        mild_seed = np.random.uniform()
        delimiters = (',', ' ', '.')
        if mild_seed >= (2/3):
            order = [1, 0, 2, 3]
        elif mild_seed >= (1/3):
            order = [0, 1, 3, 2]
        else:
            order = [1, 0, 3, 2]

        new_row = np.array([row[i] for i in order])
        address = self._concat_row(new_row)
        result = self._permutate_delimiters(address, delimiters)

        return result

    @staticmethod
    def _permutate_delimiters(address_string, delimiters):
        new_letter_map = map(
            lambda x: choice(delimiters) + ' ' if x in {',', ' '} else x,
            address_string
            )

        result = reduce(lambda x, y: x+y, new_letter_map)

        return result

    @staticmethod
    def _concat_row(row):
        result = f'{row[0]} {row[1]}, {row[2]} {row[3]}'.lower() \
            .replace(' , ', ', ')

        return result

    @staticmethod
    def _assign_bools(source, maxlen, codec):
        target = np.zeros(
            shape=(len(source), maxlen, len(codec)),
            dtype=bool
            )
        for i, sentence in enumerate(source):
            for t, letter in enumerate(sentence):
                target[i, t, codec[letter]] = True

        return target