from pathlib import Path
from luigi.util import inherits
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.build_features import PrepareHeartDataset


class DNNParameters(luigi.Config):
    # Simple config class to hold some globals and config parameters

    untrained_model_path = luigi.Parameter(Path('models/heart_dnn_clf_model_untrained.hdf5'))

    trained_model_path = luigi.Parameter(Path('models/heart_dnn_clf_model_trained.hdf5'))
    training_plot_path = luigi.Parameter(Path('models/heart_dnn_clf_training.pdf'))

    eval_results_path = luigi.Parameter(Path('models/heart_dnn_evaluation.csv'))


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

        # Load training data
        x_train = pd.read_csv(PrepareHeartDataset.train_x_path)
        y_train = pd.read_csv(PrepareHeartDataset.train_y_path)
        y_train_cat = keras.utils.to_categorical(y_train)

        # Load untrained model
        model = keras.models.load_model(self.untrained_model_path)

        # Fit model to data
        history = model.fit(x_train,
                            y_train_cat,
                            batch_size=40,
                            epochs=500,
                            validation_split=0.1)

        # Plot training progress
        self.make_training_plot(history.history,
                                self.training_plot_path)

        # Store trained model to disk
        model.save(self.trained_model_path)

    @staticmethod
    def make_training_plot(history_dict,
                           path):
        """Produces and stores a plot showing the training progress from the
           history-dict produced while training a DNN.

        Parameter
        ---------
        history_dict : dict
            Dictionary with keys: loss, accuracy, val_loss and val_accuracy
        path : Path or string
            Path to the location
        """
        fig = plt.figure(tight_layout=True)
        ax_loss = fig.add_subplot(2, 1, 1)
        ax_met = fig.add_subplot(2, 1, 2)

        epochs = np.arange(len(history_dict['loss']))
        ax_loss.plot(epochs,
                     history_dict['loss'],
                     label='Loss')
        ax_loss.plot(epochs,
                     history_dict['val_loss'],
                     label='Validation Loss')

        ax_met.plot(epochs,
                    history_dict['accuracy'],
                    label='Accuracy')
        ax_met.plot(epochs,
                    history_dict['val_accuracy'],
                    label='Validation Accuracy')

        ax_loss.grid(True)
        ax_met.grid(True)

        ax_loss.set_xlabel('Epochs')
        ax_met.set_xlabel('Epochs')

        ax_loss.set_ylabel('Loss')
        ax_met.set_ylabel('Accuracy')

        ax_loss.legend()
        ax_met.legend()

        fig.savefig(path)

        return


@inherits(DNNParameters)
class EvaluateDNNModel(luigi.Task):
    """Simple task to evaluate a trained model with the test data"""

    def requires(self):
        return TrainDNNModel()

    def output(self):
        return luigi.LocalTarget(self.eval_results_path)

    def run(self):
        # Load test data
        x_test = pd.read_csv(PrepareHeartDataset.test_x_path)
        y_test = pd.read_csv(PrepareHeartDataset.test_y_path)
        y_test_cat = keras.utils.to_categorical(y_test)

        # Load trained model
        model = keras.models.load_model(self.trained_model_path)

        # Evaluate
        eval_results = model.evaluate(x_test,
                                      y_test_cat)

        # Store evaluation results on disk
        eval_df = pd.DataFrame({'loss': [eval_results[0]],
                                'accuracy': [eval_results[1]]})
        eval_df.to_csv(self.eval_results_path,
                       index=False)

        return


@inherits(DNNParameters)
class DNNModelPredict(luigi.Task):
    """Simple task to predict class on a given dataset"""

    data_for_prediction_path = luigi.Parameter(Path('data/processed/heart_predict.csv'))
    results_path = luigi.Parameter(Path('reports/results/heart_prediction_results.csv'))

    def requires(self):
        return TrainDNNModel()

    def output(self):
        return luigi.LocalTarget(self.results_path)

    def run(self):
        data = pd.read_csv(self.data_for_prediction_path)

        model = keras.models.load_model(self.trained_model_path)

        predictions = np.argmax(model.predict(data), axis=1)

        print(predictions)

        predictions = pd.DataFrame(data=np.argmax(model.predict(data), axis=1),
                                   columns=['target'])

        predictions.to_csv(self.results_path,
                           index=False)


if __name__ == '__main__':
    luigi.build([DNNModelPredict()],
                local_scheduler=True,
                detailed_summary=True)
