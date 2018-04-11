import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Creating Data
X_data = np.linspace(0, 10,1000000)
noise = np.random.randn(len(X_data))

Y_data = 5*X_data + 7 + noise
dataset = pd.concat([pd.DataFrame(X_data, columns=['X']), pd.DataFrame(Y_data, columns=['Y'])], axis=1)

#Creating Model
fet_cols = tf.feature_column.numeric_column('x', shape = [1])
my_estimator = tf.estimator.LinearRegressor(feature_columns=fet_cols)


#dividing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)


#Creating functions
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_test},y_test,batch_size=4,num_epochs=1000,shuffle=False)


#Running functions created above
my_estimator.train(input_fn=input_func,steps=1000)
train_metrics = my_estimator.evaluate(input_fn=train_input_func,steps=1000)
eval_metrics = my_estimator.evaluate(input_fn=eval_input_func,steps=1000)




#PLOTTING RESULTS
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':np.linspace(0,10,10)},shuffle=False)
predictions = []# np.array([])
for x in my_estimator.predict(input_fn=input_fn_predict):
    predictions.append(x['predictions'])


dataset.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(np.linspace(0,10,10),predictions,'r')












