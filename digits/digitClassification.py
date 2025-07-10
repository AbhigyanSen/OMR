# import tensorflow as tf
# import matplotlib.pyplot as plt

# # Load the dataset (automatically downloads if not present)
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # Show the shapes
# print("Training set shape:", x_train.shape, y_train.shape)
# print("Test set shape:", x_test.shape, y_test.shape)

# # Normalize pixel values
# x_train = x_train / 255.0
# x_test = x_test / 255.0

# # Add a channel dimension (grayscale: 1 channel)
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)

# plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
# plt.title(f"Label: {y_train[0]}")
# plt.axis('off')
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Load image (grayscale)
image_path = r"D:\Projects\OMR\new_abhigyan\digits\TEST-01010.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Invert (white background, black digit like MNIST)
img_inv = cv2.bitwise_not(img)

# Resize to consistent height if needed
img_inv = cv2.resize(img_inv, (512, 64))  # Uncomment if inconsistent resolution

# --- Step 1: Grid slicing ---
# Assuming 10 equal-width digits in 1 row
num_digits = 10
height, width = img_inv.shape
cell_width = width // num_digits

digits = []
positions = []

for i in range(num_digits):
    x1 = i * cell_width
    x2 = (i + 1) * cell_width
    digit_crop = img_inv[:, x1:x2]

    # Resize to 28x28 (MNIST format)
    digit_resized = cv2.resize(digit_crop, (28, 28))
    digit_normalized = digit_resized / 255.0
    digit_input = digit_normalized.reshape(1, 28, 28, 1)

    digits.append(digit_input)
    positions.append((x1, x2))

# --- Step 2: Model loading / training (if not already trained) ---
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers

(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)

model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

# --- Step 3: Predict and annotate ---
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
predictions = []

for i, digit_input in enumerate(digits):
    pred = model.predict(digit_input, verbose=0)
    digit = np.argmax(pred)
    predictions.append(digit)

    # Draw prediction
    x1, x2 = positions[i]
    cv2.putText(img_color, str(digit), (x1 + 10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

# --- Step 4: Save result ---
output_filename = os.path.basename(image_path)
output_path = os.path.join("output", output_filename)

os.makedirs("output", exist_ok=True)
cv2.imwrite(output_path, img_color)

# --- Step 5: Print result ---
predicted_number = ''.join(map(str, predictions))
print("Predicted digits:", predicted_number)
print("Saved annotated image to:", output_path)