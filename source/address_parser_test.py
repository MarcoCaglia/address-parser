import unittest

from source.address_parser import AddressParser
from source.address_permutator


class TestAddressParser(unittest.TestCase):
    def test_runtime(self):
        parser = AddressParser(log_level=10)
        parser.fit_new_model()
        parser.load_model(country='pt')

        addresses = 
        parser.parse_addresses('')

