""" 
This code sets up a VGG16-based model for binary image classification.
It uses a pre-trained VGG16 model as a feature extractor, adds custom classification layers, 
and trains the model on images organized in directories. The model is then evaluated on 
validation data to report accuracy.
"""
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Image data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    r'e:\MachineLearning- Internship\Task 04\data\train',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')  # 'binary' for binary classification

validation_generator = validation_datagen.flow_from_directory(
    r'e:\MachineLearning- Internship\Task 04\data\validation',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')  # 'binary' for binary classification

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Validation accuracy: {test_acc:.4f}')
