import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv(r'C:\Users\MoI\Downloads\Housing.csv')

df.mainroad.replace(('yes', 'no'), (1, 0), inplace=True)
df.guestroom.replace(('yes', 'no'), (1, 0), inplace=True)
df.basement.replace(('yes', 'no'), (1, 0), inplace=True)
df.hotwaterheating.replace(('yes', 'no'), (1, 0), inplace=True)
df.airconditioning.replace(('yes', 'no'), (1, 0), inplace=True)
df.prefarea.replace(('yes', 'no'), (1, 0), inplace=True)
df.furnishingstatus.replace(('furnished', 'semi-furnished', 'unfurnished'), (2, 1, 0), inplace=True)

# print(df.head())


X = df.drop(['price', 'basement', 'hotwaterheating', 'guestroom', 'mainroad'], axis=1)
Y = df['price']
xtrain, xtest, ytrain, ytest = train_test_split(X.values, Y.values, test_size=0.25)

model = LinearRegression()
model.fit(xtrain, ytrain)

predicted = model.predict(xtest)
print(r2_score(ytest, predicted))

with open('model.pkl','wb') as files:
    pickle.dump(model,files)

