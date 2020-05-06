import logging
import pandas as pd

from address_permutator import AddressPermutator
import tensorflow as tf
from exceptions import NoModelError
from sklearn.model_selection import train_test_split


class AddressParser(AddressPermutator):
    def __init__(self, log_level=20):
        self.model = None

        LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
        logging.basicConfig(
            filename='address_parser_log.txt',
            level=log_level,
            format=LOG_FORMAT
            )

        self.logger = logging.getLogger()

    def fit_new_model(
        self, train_path, country, buckets=(0.2, 0.7, 0.1),
        no_addresses=250000, epochs=5
    ):
        """Fits model for new country.

        Arguments:
            train_path {str} -- Path to training data.
            country {str} -- ISO-2 country tag.

        Keyword Arguments:
            no_addresses {int} -- Number of addresses to use (default: {250000})
            epochs {int} -- Number of epochs (default: {5})
        """
        self.logger.info('Loading Training Data.')
        address_table = pd.read_csv(train_path).sample(no_addresses)

        const_matrix = address_table[
            ['STREET', 'NUMBER', 'POSTCODE', 'CITY']
            ].values

        X, y = self.permutate(const_matrix, buckets)

        X_train, X_test, y_train, y_test = None  # Current

    def load_resources(self, country):
        """Load resources for specified country.

        Arguments:
            country {str} -- Country for which resources should be loaded.
        """
        logging.info('Loading Model for country {country}.')
        path = f'../saved_model/{country}.h5'
        self.model = tf.keras.models.load_model(path)

    def parse_addresses(self, addresses, country=None):
        """Parse given addresses.

        Arguments:
            addresses {list} -- List of address strings to parse.

        Keyword Arguments:
            country {[type]} -- Can be used to overwrite model for
                                load_resources (default: {None}).

        Raises:
            NoModelError: Specified model was not found

        Returns:
            tuple -- Tuple of parsed address strings.
        """
        if country:
            if self.model is not None:
                self.logger.warning(
                    'Model overwritten with new country model.'
                    )
            else:
                raise NoModelError('No model loaded.')
            self.load_model(country=country)
        permutator = AddressPermutator()
        logging.info('Encoding passed addresses.')
        X_test = permutator.encode(addresses)

        y_hat = self.model.predict(X_test)
        decoded = permutator.decode(y_hat)

        return decoded
