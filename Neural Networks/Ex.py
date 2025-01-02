from keras import backend as k


def build(input_shape, classes):
    model = Sequential()

    model.add(Conv2D(20, kernel_size=5, padding="same",activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, kernel_size=5, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


NB_EPOCH = 4 # number of epochs
BATCH_SIZE = 128 # size of the batch
VERBOSE = 1 # set the training phase as verbose
OPTIMIZER = Adam() # optimizer
VALIDATION_SPLIT=0.2 # percentage of the training data used for 
evaluating the loss function
IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
NB_CLASSES = 10 # number of outputs = number of digits
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS) # shape of the input

(X_train, y_train), (X_test, y_test) = mnist.load_data()

k.set_image_dim_ordering("th")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy", 
optimizer=OPTIMIZER,metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

model.save("model2")

score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print('Test accuracy:', score[1])
