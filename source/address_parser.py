import logging
import pandas as pd

from source.address_permutator import AddressPermutator


class AddressParser:
    def __init___(self, log_level=20):
        self.model = None

        logging.basicConfig(
            filename='address_parser_log.txt',
            level=log_level
            )

        self.logger = logging.getLogger()

    def fit_new_model(self):
        pass

    def load_model(self, country):
        pass

    def parse_addresses(self, addresses, country=None):
        y_hat = self.model.predict(addresses)
