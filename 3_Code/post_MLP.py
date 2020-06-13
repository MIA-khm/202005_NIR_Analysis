import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model

data = pd.read_csv('train.csv', index_col=[0])
x_data = pd.read_csv('train_encoded_8.csv', index_col=[0])
x_data = x_data.iloc[:,71:]
y_data = data.iloc[:,71:]

input_size = 4
output_size = 4

input_layer = Input(shape=(input_size,))

x = Dense(1024, activation='relu')(input_layer)
x = Dense(512, activation='relu')(x)
prediction1 = Dense(output_size, activation='relu')(x)

model = Model(inputs = input_layer, outputs = prediction1)
model.compile(optimizer='rmsprop',
               loss='mse',
               metrics=['accuracy'])

model.fit(x_data,y_data, epochs=100, verbose=1)
df = pd.DataFrame(model.predict(x_data))
df.to_csv('pred_MLP_train_8.csv')
