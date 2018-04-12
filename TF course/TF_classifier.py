import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#data preprocessing
dataset = pd.read_csv('pima-indians-diabetes.csv')
X_data = dataset[['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps', 'Insulin', 'BMI', 'Pedigree', 'Age', 'Group']]
Y_data = dataset['Class']


#Normalizing inputs
#X_data = X_data.values
#cols = X_data.shape[1]

#for i in range(cols-2):
 #   X_data[:,i] = (X_data[:,i] - X_data[:,i].mean(axis =0)) / X_data[:, i].std(axis=0)

#Giving column names back to data after normalizing
#X_data = pd.DataFrame(data=X_data, columns=['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps', 'Insulin', 'BMI', 'Pedigree', 'Age', 'Group'])



cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps','Insulin', 'BMI', 'Pedigree']
X_data[cols_to_norm] = X_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))











#Creating Tensorflow feature columns for all columns except 'Group' column
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')


#For group column
assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

#Converting age to age buckets
age_buckets =tf.feature_column.bucketized_column(age, [20, 30, 40, 50, 60, 70, 80])

#Putting all columns together
tf_feature_columns = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, age_buckets, assigned_group]


#Train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=101)



#Input functions are used in tensorflow to input the data to model
#Inputing data and creating model
input_function = tf.estimator.inputs.pandas_input_fn(x = X_train, y= Y_train , batch_size=8, num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns= tf_feature_columns, n_classes=2)
model.train(input_fn=input_function, steps=1000)


eval_imput_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y= Y_test , batch_size=8, num_epochs=20000, shuffle=True)
model.evaluate(eval_imput_func)














































