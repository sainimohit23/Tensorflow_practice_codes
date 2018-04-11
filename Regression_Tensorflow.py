import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Creating Data
X_data = np.linspace(0, 10,1000000)
noise = np.random.randn(len(X_data))

Y_data = 5*X_data + 7 + noise
dataset = pd.concat([pd.DataFrame(X_data, columns=['X']), pd.DataFrame(Y_data, columns=['Y'])], axis=1)


#Creating variables and placeholders
batch_size = 8
X_ph = tf.placeholder(tf.float32, [batch_size])
Y_ph = tf.placeholder(tf.float32, [batch_size])

m = tf.Variable(1.0)
b = tf.Variable(0.7)


#Model
Y_hat = m*X_ph + b
cost = tf.reduce_sum(tf.squared_difference(Y_hat, Y_ph))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

init = tf.global_variables_initializer()

#Creating and Running session
with tf.Session() as sess:
    sess.run(init)
    
    batches = 20000
    
    for i in range(batches):
        indices = np.random.randint(0, len(X_data), size=batch_size)
        
        sess.run(optimizer, feed_dict={X_ph:X_data[indices], Y_ph: Y_data[indices]})
        
    m1, b1 = sess.run([m, b])
    
    
    
    
#Plotting results
dataset.sample(300).plot(x='X', y='Y', kind = 'scatter')
Y_hat = m1*X_data + b1
plt.plot(X_data, Y_hat, c='r')    
plt.show()    
