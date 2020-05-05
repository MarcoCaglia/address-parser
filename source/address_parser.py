import logging
import pandas as pd

from source.address_permutator import AddressPermutator
import tensorflow as tf


class AddressParser:
    def __init___(self, log_level=20):
        self.model = None

        LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
        logging.basicConfig(
            filename='address_parser_log.txt',
            level=log_level,
            format=LOG_FORMAT
            )

        self.logger = logging.getLogger()

    def fit_new_model(self, country=None):
        self.logger.warning('Model fitting is not currently implemented.')

    def load_model(self, country):
        logging.info('Loading Model for country {country}.')
        path = '../saved_model/{country}.h5'
        self.model = tf.keras.models.load_model(path)

    def parse_addresses(self, addresses, country=None):
        if not country:
            if self.model is not None:
                self.logger.warning(
                    'Model overwritten with new country model.'
                    )

            self.load_model(country=country)
        permutator = AddressPermutator()
        logging.info('Encoding passed addresses.')
        X_test = permutator.encode(addresses)

        y_hat = self.model.predict(X_test)
        decoded = permutator.decode(y_hat)

        return decoded
