from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, jsonify
from flask_marshmallow import Marshmallow
from sqlalchemy import create_engine
import os
import pandas as pd
from ml_model.train import train_model
from ml_model.prediction import make_prediction

# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
DATABASE_URL = 'postgres://implhxkm:j6dDMwnWFzLmx2vdu1Ckuyp4aG1NVk9l@manny.db.elephantsql.com:5432/implhxkm'
# Database
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Init db
db = SQLAlchemy(app)
# Init ma
ma = Marshmallow(app)

engine = create_engine(DATABASE_URL)

# make a predict
@app.route('/predict', methods=['POST'])
def predict():
    y_test = make_prediction(input_data=request.get_json())
    df = pd.DataFrame(request.get_json(), index=[0])
    df['Survived'] = y_test
    df.to_sql('predictions', con=engine, if_exists='append')
    return jsonify((y_test.tolist()))


@app.route('/data', methods=['GET'])
def data():
    return pd.read_sql('passengers', DATABASE_URL).to_dict(orient="index")


@app.route('/train', methods=['GET'])
def train():
    train_model()
    return "Training completed"


# Run Server
if __name__ == '__main__':
    app.run(debug=True)