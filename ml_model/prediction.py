import pickle
import pandas as pd


def make_prediction(model_path, input_data):
    if type(input_data) is dict:
        data = pd.DataFrame(input_data, index=[0])
    else:
        data = pd.DataFrame(input_data)

    loaded_model = pickle.load(open(model_path, 'rb'))
    result = loaded_model.predict(data)
    return result
