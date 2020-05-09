import unittest

import pandas as pd

from address_parser import AddressParser


class TestAddressParser(unittest.TestCase):
    def test_model_creation(self):
        parser = AddressParser(log_level=10)

        test_path = '../data/openaddr-collected-europe/pt/countrywide.csv'

        test_data = pd.read_csv(test_path).sample(100)

        test_matrix = test_data[
            ['STREET', 'NUMBER', 'POSTCODE', 'CITY']
            ].values

        test_body = map(
            lambda x: f'{x[0]} {x[1]}, {x[2]} {x[3]}'.lower(),
            test_matrix
            )

        parser.fit_new_model(
            train_path=test_path,
            country='test',
            no_addresses=100,
            testing=True
        )

        parser.load_resources(country='test', testing=True)

        parser.parse_addresses(list(test_body))
