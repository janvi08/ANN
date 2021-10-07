import pandas as pd
import numpy as np

df = pd.read_csv('../DATA/cancer_classification.csv')
df.info()
df.describe().transpose()

# ## EDA
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='benign_0__mal_1',data=df)
sns.heatmap(df.corr())

df.corr()['benign_0__mal_1'].sort_values()
df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')

df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

# ## Train Test Split

X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

# ## Scaling Data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#     # For a binary classification problem
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#                   
#     


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout

X_train.shape
model = Sequential()


model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# ## Training the Model 
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1
          )

# model.history.history
model_loss = pd.DataFrame(model.history.history)

# model_loss
model_loss.plot()

# #early stopping
model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

from tensorflow.keras.layers import Dropout


model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# # Model Evaluation
predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
