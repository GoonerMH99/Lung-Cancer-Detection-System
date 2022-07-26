from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import numpy as np

cancer = np.load("/content/drive/MyDrive/PreProcessed-Cancer-SingleSlices-pixdata(centered).npy", allow_pickle=True)
normal = np.load("/content/drive/MyDrive/PreProcessed-Normal-SingleSlices-pixdata(centered).npy", allow_pickle=True)

data = np.append(cancer, normal, axis=0)
label = [True]*len(cancer) + [False]*len(normal)

img_test = []
img_train = []
LabelTest = []
LabelTrain = []

for i in range(len(data)):
    if i % 10 == 0:
        img_test.append(data[i])
        LabelTest.append(label[i])
        continue
    img_train.append(data[i])
    LabelTrain.append(label[i])


print(LabelTrain)
outProcessTrain = preprocessing.LabelEncoder()
outProcessTest = preprocessing.LabelEncoder()

outProcessTest.fit(LabelTest)
LabelTest = outProcessTest.transform(LabelTest)

outProcessTrain.fit(LabelTrain)
LabelTrain = outProcessTrain.transform(LabelTrain)

img_train = np.array(img_train)
img_train = np.expand_dims(img_train, axis=3)
X_train = img_train.reshape(img_train.shape[0], 50, 50, 1)

img_test = np.array(img_test)
img_test = np.expand_dims(img_test, axis=3)
X_test = img_test.reshape(img_test.shape[0], 50, 50, 1)

LabelTrain = np.array(LabelTrain)
LabelTest = np.array(LabelTest)

np.save('Test.npy', X_test)
np.save('LabelTest.npy', LabelTest)


# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(25, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(50,50,1)))
# model.add(MaxPool2D(pool_size=(2,2)))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(100, activation='relu'))

# output layer
model.add(Dense(2, activation='softmax'))

# looking at the model summary
model.summary()
# compiling the sequential model
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training the model for 10 epochs
model.fit(X_train, LabelTrain, batch_size=128, epochs=10, validation_data=(X_test, LabelTest))

model.save("model(2Layers-2nodesOP-93acc).h5")
