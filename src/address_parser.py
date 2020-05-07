import json
import logging
from exceptions import NoModelError

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, LeakyReLU,
                                     TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from address_permutator import AddressPermutator


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
        no_addresses=250000, epochs=5, testing=False
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
        X, y = self.encode(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75
            )

        model = self._generate_model(X, y)

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            shuffle=True,
            batch_size=128,
            validation_split=0.15
            )

        model.evaluate(X_test, y_test)

        if not testing:
            model.save(f'./saved_model/{country}_model.h5')
        else:
            model.save(f'../saved_model/{country}_test.h5')

    def load_resources(self, country, testing=False):
        """Load resources for specified country.

        Arguments:
            country {str} -- Country for which resources should be loaded.
        """
        logging.info(f'Loading Model for country {country}.')
        if not testing:
            path = f'./saved_model/{country}_model.h5'
        else:
            path = f'../saved_model/{country}_test.h5'
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

    @staticmethod
    def _generate_model(X, y):
        logging.info('Generating new model.')

        model = Sequential()

        model.add(LSTM(
            512,
            return_sequences=True,
            input_shape=(len(X[0]), len(X[0][0]))
                ))
        model.add(LeakyReLU())

        model.add(LSTM(256, return_sequences=True))
        model.add(LeakyReLU())

        model.add(TimeDistributed(Dense(128)))
        model.add(LeakyReLU())

        model.add(TimeDistributed(Dense(len(X[0][0]), activation='softmax')))

        optimizer = Adam(lr=0.01)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
            )

        logging.info(f'Model generated. Model summary: {model.summary()}')
        model.summary()

        return model
