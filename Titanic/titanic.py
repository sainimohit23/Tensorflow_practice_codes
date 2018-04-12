import pandas as pd
import tensorflow as tf
import numpy as np

def age_imputer(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name='X')
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    
    return X, Y



def initialize_parameters():
    W1= tf.get_variable('W1', [25, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1= tf.get_variable('b1', [25, 1], initializer = tf.zeros_initializer())
    W2= tf.get_variable('W2', [25, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2= tf.get_variable('b2', [25, 1], initializer = tf.zeros_initializer())
    W3= tf.get_variable('W3', [1, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3= tf.get_variable('b3', [1, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3
    

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost




dataset = pd.read_csv("train.csv")
dataset['Age'] = dataset[['Age', 'Pclass']].apply(age_imputer, axis=1)
dataset = dataset.iloc[:, [1, 2, 4, 5, 6, 7, 9, 11]]
dataset = dataset.dropna()
X = dataset.iloc[:, 1:]
Y = dataset.iloc[:, 0].values
Y = Y.reshape(Y.shape[0], 1)




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
X.iloc[:,[0, 1, 6]] = X.iloc[:,[0, 1, 6]].apply(encoder.fit_transform)

onehot = OneHotEncoder(n_values='auto', categorical_features=[0, 1, 6])
X = onehot.fit_transform(X).toarray()
X = X.T
Y = Y.T

tf.reset_default_graph() 
n_x, m = X.shape
n_y = 1


costs = []

X_placeholder, Y_placeholder = create_placeholders(n_x, n_y)
parameters = initialize_parameters()
Z3 = forward_propagation(X_placeholder, parameters)

cost = compute_cost(Z3, Y_placeholder)
Optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
batch_X, batch_Y =X, Y
_, mb_cost = sess.run([Optimizer, cost], feed_dict= {X:batch_X, Y:batch_Y})
costs.append(mb_cost)
        













































































