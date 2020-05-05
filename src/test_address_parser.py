import unittest

import pandas as pd

from address_parser import AddressParser
from address_permutator import AddressPermutator


class TestAddressParser(unittest.TestCase):
    def test_runtime(self):
        parser = AddressParser(log_level=10)
        parser.fit_new_model()
        parser.load_model(country='pt')

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

        parser.parse_addresses(perm)
