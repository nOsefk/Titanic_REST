import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('ml_model/data/train.csv')
url = 'postgres://implhxkm:j6dDMwnWFzLmx2vdu1Ckuyp4aG1NVk9l@manny.db.elephantsql.com:5432/implhxkm'
engine = create_engine(url)
df.to_sql('passengers', con=engine)



