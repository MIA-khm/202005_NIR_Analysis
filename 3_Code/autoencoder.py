import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras import layers
from keras import regularizers
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv', index_col=[0])
test = pd.read_csv('test.csv', index_col=[0])
print(np.shape(data))

x_data = data.iloc[:,1:36] #to src
test_data = test.iloc[:,1:36]
#
#y_data = data.iloc[:,71:]

print(x_data.columns)



input_dim = np.shape(x_data)[1]
encoding_dim_list = [4,5,6,7,8,9,10]
epochs = 100
hidden = [32,16]
mse_list = []

for i in range(len(encoding_dim_list)):

    print(f'{i}/{len(encoding_dim_list)}')

    encoding_dim = encoding_dim_list[i]
    input_en = Input(shape = (input_dim,))
    encoded = Dense(hidden[0], activation= 'relu',
                    activity_regularizer=regularizers.l1(10e-9))(input_en)
    encoded = Dense(hidden[1], activation= 'relu',
                    activity_regularizer=regularizers.l1(10e-9))(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(hidden[1], activation='relu')(encoded)
    decoded = Dense(hidden[0], activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)



    autoencoder = Model(input_en, decoded)

    encoder = Model(input_en, encoded)

    #encoded_input = Input(shape=(encoding_dim,))
    #decoder_layer = autoencoder.layers[-1]
    #decoder = Model(encoded_input, decoder_layer(encoded_input))

    #plot_model(autoencoder, show_shapes=True, to_file='autoencoder.png')

    autoencoder.compile(optimizer='rmsprop', loss='mse')

    autoencoder.fit(x_data,x_data,
                    epochs=epochs,
                    shuffle=True,
                    verbose=0,
                    validation_data=[test_data,test_data])

    #pred = pd.DataFrame(autoencoder.predict(x_data))
    #pred.to_csv('autoencoder_rlt.csv')
    pred = pd.DataFrame(encoder.predict(x_data))
    pred.to_csv(f'encoded_rlt_train_{encoding_dim}.csv')
    pred = pd.DataFrame(encoder.predict(test_data))
    pred.to_csv(f'encoded_rlt_test_{encoding_dim}.csv')

    acc_pred = pd.DataFrame(autoencoder.predict(x_data))
    mse = mean_squared_error(x_data, acc_pred)

    mse_list.append(mse)


acc_df = pd.DataFrame({'num':encoding_dim_list,'mse':mse_list})
acc_df.to_csv('acc_df.csv')
