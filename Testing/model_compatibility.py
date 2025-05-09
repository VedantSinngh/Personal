interpreter = tf.lite.Interpreter(model_path='bird_detector.tflite')
interpreter.allocate_tensors()  # Will raise error if truly incompatible
print("Model loaded successfully!")