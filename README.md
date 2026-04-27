# Ecosort-
# AI Waste Classifier

Real-time waste classification using a TFLite MobileNetV3 model and OpenCV.

## Requirements

- Python 3.8+
- OpenCV (`pip install opencv-python`)
- TensorFlow Lite (`pip install tflite-runtime` or `pip install tensorflow`)
- NumPy (`pip install numpy`)
- A webcam
- `waste_classifier.tflite` model file in the project root

## Usage

python classifier.py

Point your webcam at waste items — cardboard, plastic, metal, or glass — and
place them inside the center box. Press ESC to quit.

## Configuration

Edit the top of `classifier.py`:
- `CONF_THRESHOLD` — minimum confidence to accept a prediction (default: 0.6)
- `HISTORY_SIZE` — number of frames used for temporal smoothing (default: 10)
- `CLASS_NAMES` — class labels matching your model's output order

## How it works

1. Each frame is center-cropped and preprocessed with MobileNetV3 normalization
2. The TFLite interpreter runs inference every 3rd frame (frame skipping)
3. Predictions above the confidence threshold are added to a rolling history buffer
4. The most frequent prediction in the buffer is shown (temporal smoothing)
5. A confidence bar and color-coded label overlay are drawn on the live feed
