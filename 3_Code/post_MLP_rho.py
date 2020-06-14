import pandas as pd
from keras.layers import Dense, Input, Dropout
from keras.models import Model
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from keras.backend.tensorflow_backend import clear_session
import random

file_list = [4,5,6,7,8,9,10]
acc_list = []
val_acc_list = []

input_size = 4
output_size = 4
batch_size = 10
epochs = 1000
lr = 0.001
ac_fn = 'tanh'
dr = 0.2
crt_rate = 0.2

for i in file_list:

    data = pd.read_csv('train.csv', index_col=[0])
    x_data = pd.read_csv(f'encoded_result/train_encoded_{i}.csv', index_col=[0])
    test_data = pd.read_csv(f'encoded_result/encoded_{i}.csv', index_col=[0])
    test = pd.read_csv('test.csv',index_col=[0])

    x_rho = np.array(x_data.iloc[:,0])
    x_rho = x_rho.reshape(len(x_rho),1)

    enc = OneHotEncoder()
    enc.fit(x_rho)

    x_rho_enc = enc.transform(x_rho).toarray()

    x_data = x_data.iloc[:,71:]
    y_data = data.iloc[:,71:]


    crt = round(np.shape(x_data)[0] * crt_rate)
    crt_val = random.sample(range(np.shape(x_data)[0]),crt)
    crt_train = np.delete(range(np.shape(x_data)[0]),crt_val)
    x_train = x_data.iloc[crt_train,:]
    x_rho_train = x_rho_enc[crt_train]
    y_train = y_data.iloc[crt_train,:]

    x_val = x_data.iloc[crt_val,:]
    x_rho_val = x_rho_enc[crt_val]
    y_val = y_data.iloc[crt_val,:]




    test_rho = test.iloc[:,0]
    test_rho = np.array(test.iloc[:,0])
    test_rho = test_rho.reshape(len(test_rho),1)

    test_rho_enc = enc.transform(test_rho).toarray()





    input_layer = Input(shape=(input_size,))
    input_rho = Input(shape=(4,))
    rho_layer = Dense(512, activation='tanh')(input_rho)
    #rho_layer = Dropout(dr)(rho_layer)
    #rho_layer = Dense(256, activation='tanh')(rho_layer)
    #rho_layer = Dropout(dr)(rho_layer)
    rho_con = Dense(256, activation='tanh')(rho_layer)

    x = Dense(512, activation=ac_fn)(input_layer)
    con = Dense(256 , activation=ac_fn)(x)
    main = keras.layers.concatenate([con,rho_con])
    main = Dropout(dr)(main)
    #main = Dense(2048, activation=ac_fn)(main)
    #main = Dropout(dr)(main)
    main = Dense(2048, activation=ac_fn)(main)
    main = Dropout(dr)(main)
    main = Dense(1024, activation=ac_fn)(main)
    main = Dropout(dr)(main)
    main = Dense(512, activation=ac_fn)(main)
    prediction1 = Dense(output_size, activation='relu')(main)

    opt = keras.optimizers.RMSprop(lr=lr)

    model = Model(inputs = [input_layer,input_rho], outputs = prediction1)
    model.compile(optimizer=opt,
                   loss='mae',
                   metrics=['accuracy'])

    model.fit([x_train,x_rho_train],y_train, epochs=epochs, verbose=1, validation_data=[[x_val,x_rho_val], y_val])
    df = pd.DataFrame(model.predict([test_data,test_rho_enc]))


    df.columns = test_data.columns
    df.index = test_data.index

    df.to_csv(f'pred_MLP_train_{i}.csv')

    train_pred = pd.DataFrame(model.predict([x_train,x_rho_train]))
    val_pred = pd.DataFrame(model.predict([x_val, x_rho_val]))

    acc_list.append(mean_absolute_error(train_pred,y_train))
    val_acc_list.append(mean_absolute_error(val_pred,y_val))

    print(mean_absolute_error(val_pred,y_val))
    clear_session()

acc_df = pd.DataFrame({'feature num':file_list, 'train acc':acc_list,'val acc':val_acc_list})
acc_df.to_csv('acc_post_MLP.csv')