"""
Batch Client for Fashion Item Classification Service

This script simulates the automated nightly job that processes newly arrived items
for classification. It selects random images, encodes them to Base64, sends
them as a batch to the API service, and displays the classification results.

This demonstrates the complete workflow of our MLOps service for refund item processing.
"""

import os
import random
import base64
import json
import requests
from typing import List, Dict, Any
from pathlib import Path


def get_random_images(test_dir: str, num_images: int = 8) -> List[Dict[str, str]]:
    """
    Select random images from the available dataset to simulate newly arrived items.

    Args:
        test_dir (str): Path to the images directory
        num_images (int): Number of images to select for the batch

    Returns:
        List[Dict]: List of dictionaries containing file paths and filenames
    """
    print(f"Scanning inventory directory: {test_dir}")

    # Get all subdirectories (category folders)
    category_dirs = [d for d in os.listdir(test_dir)
                     if os.path.isdir(os.path.join(test_dir, d))]

    print(
        f"Found {len(category_dirs)} item categories: {', '.join(category_dirs)}")

    # Collect all image files from all category directories
    all_images = []
    for category_name in category_dirs:
        category_path = os.path.join(test_dir, category_name)
        image_files = [f for f in os.listdir(category_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            all_images.append({
                'path': os.path.join(category_path, image_file),
                'filename': image_file
            })

    print(f"Total images available: {len(all_images)}")

    # Randomly select the specified number of images
    selected_images = random.sample(
        all_images, min(num_images, len(all_images)))

    print(f"Selected {len(selected_images)} images for batch processing")
    return selected_images


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to Base64 string for API transmission.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Base64-encoded image data
    """
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
            return base64_string
    except Exception as e:
        raise Exception(f"Failed to encode image {image_path}: {str(e)}")


def send_batch_request(images_data: List[Dict], api_url: str) -> Dict[str, Any]:
    """
    Send a batch of Base64-encoded images to the classification API.

    Args:
        images_data (List[Dict]): List of image information including Base64 data
        api_url (str): URL of the prediction API endpoint

    Returns:
        Dict: API response containing predictions
    """
    print(f"\nSending batch request to API: {api_url}")

    # Prepare the request payload
    payload = {
        "images": [img['base64_data'] for img in images_data]
    }

    try:
        # Send POST request to the API
        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30  # 30 second timeout
        )

        # Check if request was successful
        if response.status_code == 200:
            print("✓ Batch processing completed successfully")
            return response.json()
        else:
            print(
                f"✗ API request failed with status code: {response.status_code}")
            print(f"Error response: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect to API server")
        print("  Make sure the Flask server is running: python api/app.py")
        return None
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return None


def display_results(images_data: List[Dict], api_response: Dict[str, Any]) -> None:
    """
    Display the classification results in a clean, table-like format.
    Also includes performance metrics for monitoring as described in slide 11.

    Args:
        images_data (List[Dict]): Original image data with filenames
        api_response (Dict): API response containing predictions
    """
    print("\n" + "="*70)
    print("FASHION ITEM CLASSIFICATION RESULTS")
    print("="*70)

    predictions = api_response.get('predictions', [])
    batch_size = api_response.get('batch_size', 0)
    processing_time = api_response.get('processing_time_ms', 0)

    print(f"Processed {batch_size} items in this batch")
    print(f"Total processing time: {processing_time:.2f}ms ({processing_time/batch_size:.2f}ms per item)\n")

    # Table header
    print(f"{'#':<3} {'Filename':<25} {'Predicted Class':<15} {'Confidence':<12}")
    print("-" * 70)

    # Results for each image
    for i, (img_data, prediction) in enumerate(zip(images_data, predictions)):
        filename = img_data['filename']
        predicted_class = prediction['predicted_class']
        confidence = prediction['confidence']

        # Format confidence as percentage
        confidence_pct = f"{confidence:.1%}"

        print(f"{i+1:<3} {filename:<25} {predicted_class:<15} {confidence_pct:<12}")

    print("-" * 70)
    print(f"Batch processing completed for {len(predictions)} items")
    print("="*70)


def main():
    """
    Main function that orchestrates the batch processing workflow.
    Demonstrates the key benefits of our batch processing approach:
    - Processes multiple items at once (higher throughput)
    - Utilizes efficient batched model inference
    - Designed for scheduled execution (simulating overnight processing)
    """
    print("="*60)
    print("FASHION ITEM BATCH CLASSIFICATION CLIENT")
    print("Simulating automated nightly processing of refund items")
    print("="*60)

    # Configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(project_root, 'data', 'test')
    # Updated port to avoid MLflow conflict
    api_url = 'http://127.0.0.1:5001/predict'
    batch_size = 32

    # Step 1: Select random images (simulating newly arrived items)
    print(f"\nSTEP 1: Selecting {batch_size} random items from inventory...")
    try:
        selected_images = get_random_images(test_dir, batch_size)
    except Exception as e:
        print(f"Error selecting images: {e}")
        return

    # Step 2: Encode images to Base64 for API transmission
    print(f"\nSTEP 2: Preparing images for transmission...")
    for img_data in selected_images:
        try:
            img_data['base64_data'] = encode_image_to_base64(img_data['path'])
            print(f"  ✓ Encoded: {img_data['filename']}")
        except Exception as e:
            print(f"  ✗ Failed to encode: {img_data['filename']} - {e}")
            return

    # Step 3: Send batch request to classification API
    print(f"\nSTEP 3: Sending batch to classification service...")
    api_response = send_batch_request(selected_images, api_url)

    if api_response is None:
        print("\n❌ Batch processing failed. Please check the API service.")
        return
    batch_id = api_response.get('batch_id')
    if batch_id:
        print(f"\nServer assigned batch_id: {batch_id}")
    # Step 4: Display results in a clean format
    print(f"\nSTEP 4: Formatting prediction results...")
    display_results(selected_images, api_response)

    print(f"\nBatch processing workflow completed successfully!")
    print("Classification results ready for database storage and workflow routing.")


if __name__ == '__main__':
    # Set random seed for reproducible demo results (optional)
    random.seed(42)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
