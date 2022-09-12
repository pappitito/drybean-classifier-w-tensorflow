import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import RMSprop



#create callback
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('accuracy') >= 0.9):
           self.model.stop_training = True

#function for classifying bean input
def classifier(my_array):
    print(my_array)
    for item in my_array:
      print(max(item))
      if (item[0] == max(item)):
        print('your classification of beans is DERMASON')
      if (item[1] == max(item)):
        print('your classification of beans is SIRA')
      if (item[2] == max(item)):
        print('your classification of beans is SEKER')
      if (item[3] == max(item)):
        print('your classification of beans is HOROZ')
      if (item[4] == max(item)):
        print('your classification of beans is CALI')
      if (item[5] == max(item)):
        print('your classification of beans is BARBUNYA')
      if (item[6] == max(item)):
        print('your classification of beans is BOMBAY')

#after tokenizing this is function to make first token begin from zero
def subtract_one(my_array):
    value = 0
    new_array =[]
    for row in my_array:
        for value in row:
            value = value - 1
            new_array.append([value]) 
          
    return new_array


  
        
    
    
        

#get dataset file which is in excel format
myfile = pd.read_excel(r'/Users/mac/Documents/work life/programming/machine learning/datasets/DryBeanDataset/Dry_Bean_Dataset.xlsx')

new_file = np.asarray(myfile)
#training data based on three parameters(Perimeter, MajorAxisLength, MinorAxisLength)
training_data = new_file[:,1:-13]

training_label = new_file[:,-1]

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(training_label)
training_sequences = tokenizer.texts_to_sequences(training_label)
print(tokenizer.word_index)
new_labels = training_sequences
print(training_data.shape)
new_labels = subtract_one(new_labels)
training_data = np.array(training_data).astype(np.float32)
new_labels = np.array(new_labels)






    
callback = mycallback()


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (3,)),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])

model.fit(training_data,new_labels, epochs = 100, callbacks=[callback])

beans = [[1279.356, 451.3612558, 323.7479961]]
beans2 = [[856.204, 331.6249644, 198.2840686]]
history = model.predict(beans)
history2 = model.predict(beans2)

classifier(history)
classifier(history2)


    





