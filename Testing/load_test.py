import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path='bird_detector.tflite')
print("Input details:", interpreter.get_input_details())
print("Output details:", interpreter.get_output_details())