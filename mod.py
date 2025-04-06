import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# Define dataset paths
train_dir = "fer13/train"
test_dir = "fer13/test"

# Data Augmentation with more transformations to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2  # Changed to standard 80/20 split
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load Training Dataset with increased image size
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(48, 48),  # Using standard FER2013 image size
    batch_size=64,  # Increased batch size
    color_mode="grayscale",
    class_mode="categorical", 
    shuffle=True, 
    subset="training"
)

# Load Validation Dataset
val_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(48, 48), 
    batch_size=64, 
    color_mode="grayscale",
    class_mode="categorical", 
    shuffle=True, 
    subset="validation"
)

# Load Testing Dataset
test_generator = test_datagen.flow_from_directory(
    test_dir, 
    target_size=(48, 48), 
    batch_size=64, 
    color_mode="grayscale",
    class_mode="categorical"
)

# Print the classes detected to ensure all are loaded
print("Classes found:", train_generator.class_indices)
num_classes = len(train_generator.class_indices)
print(f"Total number of classes: {num_classes}")

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Enhanced CNN Model with BatchNormalization and more capacity
model = Sequential([
    # First convolutional block
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Second convolutional block
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Third convolutional block
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Flatten and dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Dynamically set based on classes
])

# Use Adam optimizer with a more appropriate learning rate
optimizer = Adam(learning_rate=0.0005)

# Compile Model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add callbacks to improve training
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Print model summary
model.summary()

# Train Model with significantly more epochs and callbacks
history = model.fit(
    train_generator,
    epochs=50,  # Increased epochs with early stopping
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stopping]
)

# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save the model
model.save("emotion_detection_model_improved.h5")

# For testing on a few examples, visualize confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get predictions for the test set
predictions = []
true_labels = []

# Reset the generator
test_generator.reset()

# Get a batch of test data
for i in range(min(10, len(test_generator))):  # Limit to 10 batches for quick test
    x, y = next(test_generator)
    y_pred = model.predict(x)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y, axis=1)
    
    predictions.extend(y_pred_classes)
    true_labels.extend(y_true_classes)

# Create confusion matrix
cm = confusion_matrix(true_labels, predictions)
class_names = list(train_generator.class_indices.keys())

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=class_names))

print("✅ Model trained and saved successfully!")

# Function to test on a single image
def predict_emotion(image_path):
    from tensorflow.keras.preprocessing import image
    
    img = image.load_img(image_path, color_mode="grayscale", target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)[0]
    emotion_index = np.argmax(prediction)
    emotion_label = list(train_generator.class_indices.keys())[emotion_index]
    confidence = prediction[emotion_index]
    
    print(f"Predicted emotion: {emotion_label} with {confidence:.2f} confidence")
    print("All emotions probabilities:")
    for i, emotion in enumerate(train_generator.class_indices.keys()):
        print(f"{emotion}: {prediction[i]:.4f}")
    
    return emotion_label, confidence, prediction

# Example: Uncomment the following line to test on a specific image
# predict_emotion("path_to_your_test_image.jpg")