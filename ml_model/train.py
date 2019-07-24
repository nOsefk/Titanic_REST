import pandas as pd
import os
from ml_model.pipeline_preparation import prediction_pipepline
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pickle

TRAINING_DATA_FILE = 'postgres://implhxkm:j6dDMwnWFzLmx2vdu1Ckuyp4aG1NVk9l@manny.db.elephantsql.com:5432/implhxkm'

MODEL_PATH = os.path.join(os.path.dirname(__file__)) + '/saved_models/model.sav'

engine = create_engine(TRAINING_DATA_FILE)


def _save_model(model):
   pickle.dump(model, open(MODEL_PATH, 'wb'))



def train_model():
    data = pd.read_sql_table("passengers", con=engine)
    data.dropna(subset=['Survived'], inplace=True)
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    prediction_pipepline.fit(X_train, y_train)
    print(prediction_pipepline.score(X_test, y_test))
    _save_model(prediction_pipepline)


if __name__ == '__main__':
    print(train_model())
