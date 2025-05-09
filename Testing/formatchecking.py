import os

def check_tflite(filepath):
    with open(filepath, 'rb') as f:
        content = f.read(100)  # Read first 100 bytes
    
    # Check if 'TFL3' appears in first 20 bytes
    if b'TFL3' in content[:20]:
        print("✅ VALID TFLite file (magic number found)")
        print(f"Magic number position: {content.find(b'TFL3')}")
        print(f"File size: {os.path.getsize(filepath)/1024:.1f} KB")
    else:
        print("❌ INVALID TFLite file")
        print(f"First 20 bytes (hex): {content[:20].hex()}")

check_tflite('bird_detector.tflite')