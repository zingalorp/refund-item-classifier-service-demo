"""
Flask API Service for Fashion Item Classification

This API serves a PyTorch model for classifying fashion items into 10 categories:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

The service accepts batches of Base64-encoded images and returns classification probabilities.
Designed for automated batch processing of refund items in an e-commerce system.
"""

from src import config
from src.model import ClassificationResNetSE
import os
import sys
import base64
import io
from typing import List, Dict, Any
import json
import hashlib

import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Initialize Flask application
app = Flask(__name__)

# Define class names for Fashion-MNIST dataset - MUST match ImageFolder alphabetical order
# ImageFolder loads classes alphabetically, which is the order the model was trained with
CLASS_NAMES = [
    "Ankle_boot", "Bag", "Coat", "Dress", "Pullover",
    "Sandal", "Shirt", "Sneaker", "T-shirt_top", "Trouser"
]

# Global variable to hold the loaded model
model = None

# Confidence threshold for low-confidence item detection
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# SHA256 hash of the model file, for demo alignment with slides
MODEL_SHA256 = None


def load_model() -> ClassificationResNetSE:
    """
    Load the pre-trained PyTorch model from disk.

    Returns:
        ClassificationResNetSE: The loaded model in evaluation mode
    """
    global model

    print("Loading Fashion-MNIST classification model...")

    # Initialize model architecture with correct number of classes
    model = ClassificationResNetSE(num_classes=len(CLASS_NAMES))

    # Load the trained weights from the saved model file
    model_path = os.path.join(project_root, 'models', 'fashion_mnist_v1.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load weights and handle potential device compatibility issues
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    # Compute hash once for reproducibility demo
    global MODEL_SHA256
    try:
        with open(model_path, 'rb') as f:
            MODEL_SHA256 = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        MODEL_SHA256 = 'unavailable'

    # Set model to evaluation mode (disables dropout, batch norm training mode)
    model.eval()

    print(f"Model loaded successfully from: {model_path}")
    return model


def get_inference_transforms() -> transforms.Compose:
    """
    Create the same preprocessing transforms used during training.
    Critical: Must match exactly the transforms used for model training.

    Returns:
        transforms.Compose: Preprocessing pipeline for inference
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        # Resize to 28x28
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),  # Convert PIL Image to tensor [0,1]
        # Normalize to [-1,1] range
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def decode_and_preprocess_image(base64_string: str) -> torch.Tensor:
    """
    Decode a Base64-encoded image and apply preprocessing transforms.

    Args:
        base64_string (str): Base64-encoded image data

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model inference
    """
    try:
        # Decode Base64 string to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Apply the same preprocessing transforms used during training
        transform = get_inference_transforms()
        image_tensor = transform(image)

        return image_tensor

    except Exception as e:
        raise ValueError(f"Failed to decode and preprocess image: {str(e)}")


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint with API information.
    """
    return jsonify({
        "service": "Fashion Item Classification API",
        "version": "1.0",
        "status": "running",
        "model_loaded": model is not None,
        "available_endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /predict": "Batch image classification"
        },
        "usage": {
            "predict": "POST to /predict with JSON: {'images': ['base64_string1', 'base64_string2', ...]}"
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_sha256": MODEL_SHA256,
    })


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Fashion Item Classification API",
        "model_sha256": MODEL_SHA256,
    })


@app.route('/predict', methods=['POST'])
def predict_batch():
    """
    Main prediction endpoint for batch classification of fashion items.

    Expects JSON payload with format:
    {
        "images": ["base64_encoded_image1", "base64_encoded_image2", ...]
    }

    Returns JSON response with format:
    {
        "predictions": [
            {
                "probabilities": [0.1, 0.05, 0.8, ...],
                "predicted_class": "Dress",
                "confidence": 0.8
            },
            ...
        ],
        "batch_size": 2,
        "processing_time_ms": 120.5,
        "metrics": {
            "avg_confidence": 0.85,
            "low_confidence_count": 2
        }
    }
    """
    try:
        # Start timing for performance metrics
        import time
        start_time = time.time()
        
        # Parse JSON request
        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({"error": "Request must contain 'images' field"}), 400

        base64_images = data['images']
        if not isinstance(base64_images, list) or len(base64_images) == 0:
            return jsonify({"error": "Images must be a non-empty list"}), 400

        import uuid
        batch_id = str(uuid.uuid4())
        print(f"[batch_id={batch_id}] Processing batch of {len(base64_images)} images...")

        # Decode and preprocess all images in the batch
        processed_tensors = []
        for i, base64_img in enumerate(base64_images):
            try:
                tensor = decode_and_preprocess_image(base64_img)
                processed_tensors.append(tensor)
            except Exception as e:
                return jsonify({
                    "error": f"Failed to process image {i}: {str(e)}"
                }), 400

        # Stack individual tensors into a single batch tensor
        # Shape: (batch_size, 1, 28, 28)
        batch_tensor = torch.stack(processed_tensors)

        # Perform inference on the entire batch (efficient single forward pass)
        with torch.no_grad():  # Disable gradient computation for efficiency
            logits = model(batch_tensor)  # Raw model outputs
            # Convert to probabilities
            probabilities = F.softmax(logits, dim=1)

        # Process results for each image in the batch
        predictions = []
        for i in range(len(base64_images)):
            # Extract probabilities for this image
            image_probs = probabilities[i].cpu().numpy().tolist()

            # Find the class with highest probability
            predicted_class_idx = torch.argmax(probabilities[i]).item()
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = image_probs[predicted_class_idx]

            predictions.append({
                "probabilities": image_probs,
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4)
            })

        # Calculate monitoring metrics as described in slide 11
        confidences = [p["confidence"] for p in predictions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        low_confidence_count = sum(1 for c in confidences if c < CONFIDENCE_THRESHOLD)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        print(f"[batch_id={batch_id}] Batch processing complete. Results ready.")
        print(f"[batch_id={batch_id}] Avg confidence: {avg_confidence:.2f}, Low confidence items: {low_confidence_count}")

        return jsonify({
            "batch_id": batch_id,
            "predictions": predictions,
            "batch_size": len(base64_images),
            "processing_time_ms": round(processing_time_ms, 2),
            "metrics": {
                "avg_confidence": round(avg_confidence, 4),
                "low_confidence_count": low_confidence_count,
                "confidence_threshold": CONFIDENCE_THRESHOLD
            }
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with informative message."""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/health", "/predict"]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with informative message."""
    return jsonify({
        "error": "Internal server error",
        "message": "Please check server logs for details"
    }), 500


if __name__ == '__main__':
    print("="*60)
    print("Fashion Item Classification API")
    print("="*60)

    try:
        # Load the model at startup
        model = load_model()

        print(f"Model architecture: {model.__class__.__name__}")
        print(f"Number of classes: {len(CLASS_NAMES)}")
        print(f"Classes: {', '.join(CLASS_NAMES)}")
        print("\nAPI Endpoints:")
        print("  GET  /health  - Health check")
        print("  POST /predict - Batch image classification")
        print("\nStarting Flask development server...")
        print("="*60)

        # Start the Flask development server
        # For production, use a proper WSGI server like Gunicorn
        app.run(
            host='127.0.0.1',  # Only accept local connections
            port=5001,         # Use port 5001 to avoid MLflow conflict
            debug=False        # Disable debug mode for cleaner output
        )

    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        sys.exit(1)
