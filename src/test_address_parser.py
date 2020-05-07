import unittest

import pandas as pd

from address_parser import AddressParser
from address_permutator import AddressPermutator


class TestAddressParser(unittest.TestCase):
    def test_model_training(self):
        parser = AddressParser(log_level=10)
        parser.fit_new_model(
            train_path=(
                '/home/marco/Documents/repositories/MarcoCaglia/'
                'address-parser/data/openaddr-collected-europe/pt/'
                'countrywide.csv'
                ),
            country='pt',
            no_addresses=500,
            epochs=5,
            testing=True
        )

    def test_runtime(self):
        parser = AddressParser(log_level=10)

        addresses = pd.read_csv(
            '../data/openaddr-collected-europe/pt/countrywide.csv'
            ).sample(50)
        addresses = addresses[['STREET', 'NUMBER', 'POSTCODE', 'CITY']].values

        permutator = AddressPermutator()
        perm = permutator.permutate(
            addresses,
            for_training=False,
            permutation_buckets=(0.25, 0.60, 0.15)
            )

        parser.load_resources(country='pt', testing=True)
        parser.parse_addresses(perm)
