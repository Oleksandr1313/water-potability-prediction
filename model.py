import tensorflow as tf
import pandas as pd
import numpy as np

csv_file = r"C:\Main_Folder\machine_learning\water_quality\water_potability.csv"
data = pd.read_csv(csv_file)

# Data preprocessing
properties = data.columns.to_list()
for i in properties:
    mean_val = data[i].mean()
    data[i] = data[i].fillna(mean_val)

x = data[properties[1:9]]
y = data[properties[-1]]

validation_length = int(0.8 * len(data))
x_train, x_test = x[:validation_length], x[validation_length:]
y_train, y_test = y[:validation_length], y[validation_length:]

# Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(8,)),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

# Make predictions on the test set
predictions = model.predict(x_test)

# Display the predicted labels
predicted_classes = np.round(predictions).flatten() 

# Display some predictions
for i in range(5):
    print(f"Actual Potability: {y_test.iloc[i]}, Predicted Potability: {predicted_classes[i]}")