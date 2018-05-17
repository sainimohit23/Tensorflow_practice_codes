#This model got maximum accuracy of 84.25%




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
    
def age_leveler(age):
    if age<20:
        return 9
    elif age<30 and age>=20:
        return 24
    elif age<40 and age >= 30:
        return 35
    else:
        return 50
    
        

dataset = pd.read_csv("train.csv")
dataset['Age'] = dataset[['Age', 'Pclass']].apply(age_imputer, axis=1)
dataset['Age'] = dataset['Age'].apply(age_leveler)
dataset = dataset.iloc[:, [1, 2, 4, 5, 6, 7, 9, 11]]
dataset = dataset.dropna()
X_train = dataset.iloc[:, 1:]
Y_train = dataset.iloc[:, 0].values
Y_train = Y_train.reshape(Y_train.shape[0], 1)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X_train.iloc[:, 1] = encoder.fit_transform(X_train.iloc[:, 1])
X_train.iloc[:, 6] = encoder.fit_transform(X_train.iloc[:, 6])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0, 1, 3, 4, 6])
X_train = ohe.fit_transform(X_train).toarray()



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)





def create_placeholders(n_f, n_l):
     
    X = tf.placeholder(tf.float32, shape=[None, n_f])
    Y = tf.placeholder(tf.float32, shape=[None, n_l])
    
    return X, Y

def initialize_parameters():
    W1 = tf.get_variable('W1', shape=[16, 24], initializer= tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[16, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', shape=[16, 16], initializer= tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', shape=[16, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', shape=[1, 16], initializer= tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', shape=[1, 1], initializer=tf.zeros_initializer())   

    parameters = {'W1': W1,
                  'W2': W2,
                  'b1': b1,
                  'b2': b2,
                  'W3': W3,
                  'b3': b3
                  }
    
    return parameters


def forward_prop(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    
    Z1 = tf.matmul(W1, tf.transpose(X))
    A1 = tf.nn.relu(tf.add(Z1, b1))
    Z2 = tf.matmul(W2, A1)
    A2 = tf.nn.relu(tf.add(Z2, b2))
    Z3 = tf.matmul(W3, A2)
    A3 = tf.sigmoid(tf.add(Z3, b3))
    
    return A3


def compute_cost(Z3, Y, m):
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.transpose(Y), logits=Z3))
    return cost


tf.reset_default_graph()
m, n_f = X_train.shape
n_l = Y_train.shape[1]


X, Y = create_placeholders(n_f, n_l)
parameters = initialize_parameters()
Z3 = forward_prop(X, parameters)

cost = compute_cost(Z3, Y,m)

optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()


permuts = list(np.random.permutation(m))
X_shuffle = X_train[permuts, :]
Y_shuffle = Y_train[permuts, :]


minibatch_size = 64
num_epochs = 20
costs = []
accuracies = []


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(230):
        num_minibatches = int(m/minibatch_size)
        
        for k in range(num_minibatches):
            
            X_batch = X_shuffle[k*minibatch_size:(k+1)*minibatch_size, :]
            Y_batch = Y_shuffle[k*minibatch_size:(k+1)*minibatch_size, :]
            
            
            _, mb_cost = sess.run([optimizer, cost], feed_dict={X: X_batch, Y:Y_batch})                
            
            if k%63 == 0:
                costs.append(mb_cost)
                
        predictions = tf.greater(Z3, 0.5)
        class_label = tf.cast(predictions, tf.int32)
        l = sess.run([class_label], feed_dict={X: X_train, Y:Y_train})
        l = np.array(l)
        l = l.reshape(Y_train.shape)
        accuracy = np.sum(np.equal(Y_train, l))/889
        accuracies.append(accuracy)

