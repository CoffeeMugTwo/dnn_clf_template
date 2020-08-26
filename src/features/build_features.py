# Std python libs
from pathlib import Path

# Libs from other packages
import luigi
import pandas as pd
from sklearn.model_selection import train_test_split


class PrepareHeartDataset(luigi.Task):
    """Split dataset into train and test sample and features and labels.
       Should have been written in a way, that it can be used as a preprocessing
       task for new data (i.e. data for prediction)."""

    raw_data_path = Path('data/raw/heart.csv')

    meta_data_path = Path('data/processed/heart_meta_data.csv')

    train_x_path = Path('data/processed/heart_x_train.csv')
    train_y_path = Path('data/processed/heart_y_train.csv')
    test_x_path = Path('data/processed/heart_x_test.csv')
    test_y_path = Path('data/processed/heart_y_test.csv')

    def requires(self):
        return None

    def output(self):
        """Task run successfully when 4 csv-files have been produced:
           train_x, train_y, test_x and test_y."""
        yield luigi.LocalTarget(self.train_x_path)
        yield luigi.LocalTarget(self.train_y_path)
        yield luigi.LocalTarget(self.test_x_path)
        yield luigi.LocalTarget(self.test_y_path)
        yield luigi.LocalTarget(self.meta_data_path)

    def run(self):
        """Load the raw data, perfom splits, store to disk."""

        raw_df = pd.read_csv(self.raw_data_path)

        x = raw_df.iloc[:, :-1]
        y = raw_df.iloc[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.15)

        x_train.to_csv(self.train_x_path, index=False)
        y_train.to_csv(self.train_y_path, index=False)
        x_test.to_csv(self.test_x_path, index=False)
        y_test.to_csv(self.test_y_path, index=False)

        x_shape = 1 if len(x.shape)==1 else x.shape[1]
        y_shape = 1 if len(y.shape)==1 else y.shape[1]

        meta_data = {'x_shape': [x_shape],
                     'y_shape': [y_shape],
                     'n_train': [x_train.shape[0]],
                     'n_test': [x_test.shape[0]]}

        meta_df = pd.DataFrame(data=meta_data)

        meta_df.to_csv(self.meta_data_path, index=False)

        return


if __name__ == '__main__':
    luigi.build([PrepareHeartDataset()],
                local_scheduler=True,
                detailed_summary=True)
