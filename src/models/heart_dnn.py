import luigi
import pandas as pd
from pathlib import Path
from luigi.util import inherits

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DNNParameters(luigi.Config):
    # Simple config class to hold some globals and config parameters

    untrained_model_path = luigi.Parameter(Path('models/heart_dnn_clf_model_untrained'))

    trained_model_path = luigi.Parameter(Path('models/heart_dnn_clf_model_trained'))



@inherits(DNNParameters)
class CreateDNNModel(luigi.Task):

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget(self.model_path)

    def run(self):

        # define model
        model = keras.Sequential([
            layers.Dense(14, activation='relu', name="input"),
            layers.Dense(50, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])

        # create model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Store model
        model.save(self.model_path)


@inherits(DNNParameters)
class TrainDNNModel(luigi.Task):

    def requires(self):
        yield CreateDNNModel()
        yield PrepareDataset()

    def output(self):
        return luigi.LocalTarget(self.trained_model_path)

    def run(self):
