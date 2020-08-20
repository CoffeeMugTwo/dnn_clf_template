import luigi
import pandas as pd
from pathlib import Path
from luigi.util import inherits
import h5py



# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.features.build_features import PrepareHeartDataset


class DNNParameters(luigi.Config):
    # Simple config class to hold some globals and config parameters

    untrained_model_path = luigi.Parameter(Path('models/heart_dnn_clf_model_untrained.hdf5'))

    trained_model_path = luigi.Parameter(Path('models/heart_dnn_clf_model_trained.hdf5'))


@inherits(DNNParameters)
class CreateDNNModel(luigi.Task):

    def requires(self):
        return PrepareHeartDataset()

    def output(self):
        return luigi.LocalTarget(self.untrained_model_path)

    def run(self):

        # load dataset meta info (for the shape of x and y)
        meta_data = pd.read_csv(PrepareHeartDataset.meta_data_path)
        x_shape = meta_data['x_shape'][0]
        y_shape = meta_data['y_shape'][0]

        # define model
        model = keras.Sequential([
            layers.Dense(14, activation='relu', input_shape=(x_shape,)),
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
        model_file = h5py.File(self.untrained_model_path)
        model.save(model_file)


@inherits(DNNParameters)
class TrainDNNModel(luigi.Task):

    def requires(self):
        yield CreateDNNModel()
        yield PrepareHeartDataset()

    def output(self):
        return luigi.LocalTarget(self.trained_model_path)

    def run(self):
        """Loads the preprocessed training data and the untrained model, then
           performs the training. Stores the trained model then on disk."""

        x_train = pd.read_csv(PrepareHeartDataset.train_x_path)
        y_train = pd.read_csv(PrepareHeartDataset.train_y_path)

        print(x_train)
        print(y_train)

        model = keras.models.load_model(self.untrained_model_path)

        print(model)


if __name__ == '__main__':
    luigi.build([TrainDNNModel()],
                local_scheduler=True,
                detailed_summary=True)
