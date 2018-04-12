import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Some artificial linear data
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5,10)

#function
l,k = np.random.rand(2)
a = tf.Variable(1.1)
b = tf.Variable(1.1)

error = 0

for x,y in zip(x_data,y_label):
    
    y_hat = a*x + b  #Our predicted value
    
    error += (y-y_hat)**2
    
    
#Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

#Training
with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 100
    
    for i in range(epochs):
        
        sess.run(train)
        

    # Fetch Back Results
    final_slope , final_intercept = sess.run([a,b])
 
    
#testing
x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')

plt.plot(x_data,y_label,'*')
plt.show()

