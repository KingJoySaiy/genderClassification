import tensorflow as tf
from tensorflow import keras
import readFile
from readFile import imageH, imageW

# dictionary {id -> data}, {id -> label}
idData, idLabel = readFile.getTrainData()
idDataTest = readFile.getTestData()

my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, min_delta=0, monitor='val_loss')]

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(imageW, imageH, 1), strides=(1, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x=idData.values(),
          y=idLabel.values(),
          batch_size=32,
          epochs=30,
          verbose=1,
          callbacks=my_callbacks,
          validation_split=0.05,
          shuffle=True
          )

predictions = model.predict(idDataTest.values())

print(predictions)
