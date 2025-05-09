import os
import tensorflow as tf
from tensorflow.keras import layers, models
import time

# 1. GPU Configuration ---------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, falling back to CPU")

# 2. Dataset Loading -----------------------------------------------------------
def parse_tfrecord(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, [224, 224])  # Resize for efficiency
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    
    # For single-class classification (adjust if multi-class)
    label = tf.sparse.to_dense(example['image/object/class/label'])
    label = tf.reduce_max(label)  # Take the max class if multiple annotations
    
    return image, label

def load_dataset(tfrecord_path, batch_size=8):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return parsed_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Replace with your actual TFRecord paths
train_dataset = load_dataset(r"C:\Users\vedaa\OneDrive\Desktop\BeakyBot\Testing\Dataset\test\Birds.tfrecord")
val_dataset = load_dataset(r"C:\Users\vedaa\OneDrive\Desktop\BeakyBot\Testing\Dataset\valid\Birds.tfrecord")  # or test.tfrecord

# 3. Model Definition ----------------------------------------------------------
def create_efficient_model(input_shape=(224, 224, 3), num_classes=1):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False  # Transfer learning
    
    model = tf.keras.Sequential([
        base_model,
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    return model

model = create_efficient_model()
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Training ------------------------------------------------------------------
start_time = time.time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    steps_per_epoch=10,  # Limit training time
    validation_steps=5
)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# 5. TFLite Conversion --------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('bird_detector.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved successfully")
print(f"Model size: {len(tflite_model)/1024:.2f} KB")