"""Address parser. Parses addresses based on list of "messy" addresses.

Raises:
    NoModelError: Model for requested country was not found.

Returns:
    list -- List of parsed addresses
"""

import json
import logging
from exceptions import NoModelError
from itertools import chain

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, LeakyReLU,
                                     TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

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
            no_addresses {int} -- Number of addresses to use (default:
                                  {250000})
            epochs {int} -- Number of epochs (default: {5})
        """
        self.logger.info('Loading Training Data.')
        address_table = pd.read_csv(train_path)

        const_matrix = address_table[
            ['STREET', 'NUMBER', 'POSTCODE', 'CITY']
            ].values

        text_body = map(
            lambda x: f'{x[0]} {x[1]}, {x[2]} {x[3]}',
            const_matrix
            )
        self.encoding, self.decoding = self._create_encoding(text_body)
        self._save_encoding(
            encoding=self.encoding,
            decoding=self.decoding,
            country=country
        )
        const_matrix_train = const_matrix[
            np.random.choice(
                const_matrix.shape[0],
                no_addresses,
                replace=False
                ), :]

        X, y = self.permutate(const_matrix_train, buckets)
        X, y = self.encode(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75
            )

        model = self._generate_model(X, y)

        _ = model.fit(
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
        logging.info(f'Loading codes for country {country}.')
        self.encoding, self.decoding = self._load_codes(country)
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
            country {str} -- Can be used to overwrite model for
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
        logging.info('Encoding passed addresses.')
        X_test, _ = self.encode(addresses)

        y_hat = self.model.predict(X_test.astype(float))
        decoded = self.decode(y_hat)

        return decoded

    @staticmethod
    def _create_encoding(text_body):
        text_body = "|".join(chain.from_iterable(text_body)).lower()
        text_set = set(text_body)

        encoding = {letter: value for value, letter in enumerate(text_set)}
        decoding = {value: letter for value, letter in enumerate(text_set)}

        return encoding, decoding

    @staticmethod
    def _save_encoding(encoding, decoding, country):
        base_path = (
            '/home/marco/Documents/repositories/MarcoCaglia/address-'
            'parser/properties'
                     )
        if country not in os.listdir(base_path + '/'):
            os.mkdir(base_path + '/' + country)
        with open(f'{base_path}/{country}/encoding.json', 'w',
                  encoding='utf8') as f:
            json.dump(encoding, f, ensure_ascii=False)

        with open(f'{base_path}/{country}/decoding.json', 'w',
                  encoding='utf8') as f:
            json.dump(decoding, f, ensure_ascii=False)

    @staticmethod
    def _generate_model(X, y):
        logging.info('Generating new model.')

        model = Sequential()

        model.add(Bidirectional(LSTM(
            512,
            return_sequences=True,
            ), input_shape=(X.shape[1], X.shape[2])))
        model.add(LeakyReLU())

        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(LeakyReLU())

        model.add(TimeDistributed(Dense(128)))
        model.add(LeakyReLU())

        model.add(TimeDistributed(Dense(X.shape[2], activation='softmax')))

        optimizer = Adam(lr=0.01)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
            )

        logging.info(f'Model generated. Model summary: {model.summary()}')
        model.summary()

        return model

    @staticmethod
    def _load_codes(country):
        base_path = (
            '/home/marco/Documents/repositories/MarcoCaglia/address-'
            'parser/properties'
                     )

        with open(f'{base_path}/{country}/encoding.json', 'r') as f:
            encoding = json.load(f)

        with open(f'{base_path}/{country}/decoding.json', 'r') as f:
            decoding = json.load(f)

        return encoding, decoding
