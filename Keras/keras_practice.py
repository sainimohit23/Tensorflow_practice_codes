import keras 
import keras.backend as k
import numpy as np


train_samples = []
train_labels = []

for i in range(50):
    # The 5% of younger individuals who did experience side effects
    random_younger = np.random.randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    # The 5% of older individuals who did not experience side effects
    random_older = np.random.randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The 95% of younger individuals who did not experience side effects
    random_younger = np.random.randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    # The 95% of older individuals who did experience side effects
    random_older = np.random.randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)


model = keras.models.Sequential(layers = [keras.layers.core.Dense(16, activation='relu', input_shape=(1, )), keras.layers.core.Dense(32, activation='relu'), keras.layers.core.Dense(2, activation='softmax')])
model.compile(keras.optimizers.Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x= train_samples, y= train_labels, batch_size=10, verbose= 2, epochs=20, shuffle=True, validation_split=0.1)



#Test Data
test_labels =  []
test_samples = []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = np.random.randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    # The 5% of older individuals who did not experience side effects
    random_older = np.random.randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = np.random.randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    # The 95% of older individuals who did experience side effects
    random_older = np.random.randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)
    
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, model.predict_classes(test_samples))






"""Saving and Loading models"""
model.save('First_model.h5')

loaded_model = keras.models.load_model('First_model.h5')
loaded_model.summary()



cm = confusion_matrix(test_labels, loaded_model.predict_classes(test_samples))













