"""
Indian Bird Species Classification with EfficientNetB0 (GPU Training + TFLite Export)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from pathlib import Path

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Constants
BATCH_SIZE = 64  # Increased for GPU
TARGET_SIZE = (224, 224)
EPOCHS = 50
LEARNING_RATE = 1e-4
BASE_DIR = "data/training_set"  # Update this path

class BirdSpeciesClassifier:
    def __init__(self):
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = []
        self.history = None

    def load_data(self, data_dir):
        """Load and prepare dataset with GPU-optimized pipelines"""
        print("Loading dataset...")
        
        # Create dataset
        self._create_dataframe(data_dir)
        self._check_corrupted_images()
        self._train_test_split()
        
        # Create TensorFlow Dataset pipelines optimized for GPU
        self._create_data_pipelines()
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")

    def _create_dataframe(self, data_dir):
        """Create dataframe from image directory"""
        image_dir = Path(data_dir)
        filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + \
                   list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG'))
        labels = [os.path.split(os.path.split(x)[0])[1] for x in filepaths]
        
        self.image_df = pd.DataFrame({
            'Filepath': [str(x) for x in filepaths],
            'Label': labels
        })
        self.class_names = sorted(self.image_df['Label'].unique().tolist())

    def _check_corrupted_images(self):
        """Check for and remove corrupted images"""
        print("Checking for corrupted images...")
        corrupted = []
        for img_path in self.image_df['Filepath']:
            try:
                img = Image.open(img_path)
                img.verify()
            except (UnidentifiedImageError, OSError):
                corrupted.append(img_path)
        
        if corrupted:
            print(f"Found {len(corrupted)} corrupted images. Removing them...")
            self.image_df = self.image_df[~self.image_df['Filepath'].isin(corrupted)]
        else:
            print("No corrupted images found.")

    def _train_test_split(self):
        """Split data into train, validation and test sets"""
        print("Splitting data...")
        train_df, test_df = train_test_split(
            self.image_df,
            test_size=0.2,
            shuffle=True,
            random_state=42,
            stratify=self.image_df['Label']
        )
        self.train_df, self.val_df = train_test_split(
            train_df,
            test_size=0.2,
            shuffle=True,
            random_state=42,
            stratify=train_df['Label']
        )

    def _prepare_image(self, filepath, label):
        """GPU-optimized image preprocessing"""
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, TARGET_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    def _create_data_pipelines(self):
        """Create optimized data pipelines for GPU"""
        print("Creating GPU-optimized data pipelines...")
        
        # Training pipeline with augmentation
        train_ds = tf.data.Dataset.from_tensor_slices(
            (self.train_df['Filepath'], self.train_df['Label'])
        )
        self.train_data = train_ds.map(
            lambda x, y: self._prepare_image(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Validation pipeline
        val_ds = tf.data.Dataset.from_tensor_slices(
            (self.val_df['Filepath'], self.val_df['Label'])
        )
        self.val_data = val_ds.map(
            lambda x, y: self._prepare_image(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Test pipeline
        test_ds = tf.data.Dataset.from_tensor_slices(
            (self.test_df['Filepath'], self.test_df['Label'])
        )
        self.test_data = test_ds.map(
            lambda x, y: self._prepare_image(x, y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def build_model(self):
        """Build EfficientNetB0 model with GPU optimization"""
        print("Building model...")
        
        # GPU-optimized data augmentation
        augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
        ], name="augmentation")
        
        # Base model
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(*TARGET_SIZE, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base_model.trainable = False
        
        # Build model
        inputs = tf.keras.Input(shape=(*TARGET_SIZE, 3))
        x = augmentation(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(len(self.class_names), activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())

    def train(self):
        """Train model with GPU acceleration"""
        print("Training model on GPU...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1,
                profile_batch='500,520'
            )
        ]
        
        with tf.device('/GPU:0'):
            self.history = self.model.fit(
                self.train_data,
                validation_data=self.val_data,
                epochs=EPOCHS,
                callbacks=callbacks
            )

    def evaluate(self):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(self.test_data)
        print(f"\nTest Accuracy: {test_acc*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate predictions
        y_pred = []
        y_true = []
        for images, labels in self.test_data:
            preds = self.model.predict(images)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(labels.numpy())
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        self._plot_confusion_matrix(y_true, y_pred)

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    def export_to_tflite(self, output_path='model.tflite'):
        """Export model to TFLite format"""
        print("Exporting to TFLite...")
        
        # Convert model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # For full integer quantization (uncomment if needed)
        # converter.representative_dataset = self._representative_dataset
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model exported to {output_path}")
    
    def _representative_dataset(self):
        """Representative dataset for full integer quantization"""
        for image, _ in self.test_data.take(100):
            yield [tf.cast(image, tf.float32)]

    def save_model(self, path='bird_classifier.h5'):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")

def main():
    # Initialize classifier
    classifier = BirdSpeciesClassifier()
    
    # Load and prepare data
    classifier.load_data(BASE_DIR)
    
    # Build and train model
    classifier.build_model()
    classifier.train()
    
    # Evaluate model
    classifier.evaluate()
    
    # Save models
    classifier.save_model()
    classifier.export_to_tflite()

if __name__ == "__main__":
    main()