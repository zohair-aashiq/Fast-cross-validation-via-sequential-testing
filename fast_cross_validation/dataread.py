import pandas as pd
from sklearn.model_selection import train_test_split


class DataRead:

    def __init__(self):
        df = pd.read_csv(r"C:\Users\zohair\PycharmProjects\Fast-cross-validation-via-sequential-testing"
                         r"\sfiles\winequality-white.csv",sep=",",skipinitialspace=True)
        df_y = df.quality
        df_x = df.drop('quality', axis=1)
        self.train_x, self.test_x, self.y_train, self.test_y = train_test_split(df_x, df_y, test_size=0.2,
                                                                                random_state=42)

    def train_x(self):
        return self.train_x

    def train_y(self):
        return self.train_y
