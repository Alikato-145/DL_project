import supervision as sv
import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageOps

def detect_color_from_bbox(image, bbox, crop_ratio=0.8):
    """
    Detect average color from bounding box using advanced color analysis

    Args:
    - image: PIL Image
    - bbox: (xmin, ymin, xmax, ymax)
    - crop_ratio: Ratio of bounding box to analyze (default 0.8)

    Returns:
    - Tuple of (hue, saturation, value)
    """
    xmin, ymin, xmax, ymax = bbox
    cropped_img = image.crop((xmin, ymin, xmax, ymax))
    cropped_img = np.array(cropped_img)

    # Convert to HSV
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)

    # Apply median blur to reduce noise
    hsv_img = cv2.medianBlur(hsv_img, 5)

    # Crop center of the image
    h, w, _ = hsv_img.shape
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    y_offset, x_offset = (h - new_h) // 2, (w - new_w) // 2
    cropped_center = hsv_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w]

    # Calculate median values
    avg_hue = np.median(cropped_center[:, :, 0])
    avg_sat = np.median(cropped_center[:, :, 1])
    avg_val = np.median(cropped_center[:, :, 2])

    return avg_hue, avg_sat, avg_val

def identify_color(avg_hue, avg_sat, avg_val):
    """
    Identify color based on HSV values

    Args:
    - avg_hue: Hue value (0-180 in OpenCV)
    - avg_sat: Saturation value (0-255)
    - avg_val: Value/Brightness value (0-255)

    Returns:
    - Color name as string
    """
    # Convert OpenCV HSV to standard HSV
    hue = avg_hue * 2  # OpenCV uses 0-180, standard HSV uses 0-360
    sat = avg_sat / 255 * 100  # Convert to percentage
    val = avg_val / 255 * 100  # Convert to percentage

    # Check for white, black, or gray conditions
    if val < 20:
        return "Black"
    if sat < 20 or val > 90:
        return "White"

    # Detailed color mapping
    if (hue >= 330 or hue <= 30):  # Red
        return "Red"
    elif 30 < hue < 90:  # Yellow to Green
        if hue < 60:
            return "Yellow"
        else:
            return "Green"
    elif 90 <= hue < 150:  # Cyan to Blue
        if hue < 120:
            return "Cyan"
        else:
            return "Blue"
    elif 150 <= hue < 270:  # Blue to Purple
        if hue < 210:
            return "Blue"
        else:
            return "Purple"
    elif 270 <= hue < 330:  # Magenta to Red
        return "Red2"

    return "Unknown"

def detection(image, model):
    """
    Run object detection on an image using the YOLO model.
    Also detects colors of detected objects.
    
    Args:
        image: Input image for detection (path or Streamlit image)
        model: YOLO model instance
        
    Returns:
        Tuple of (annotated_image, color_counts)
    """
    # Handle different image input types
    if isinstance(image, str) and os.path.isfile(image):
        # If image is a file path
        img_cv2 = cv2.imread(image)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        # Also open as PIL image for color detection
        img_pil = Image.open(image)
        img_pil = ImageOps.exif_transpose(img_pil)
        img_pil = img_pil.convert("RGB")
    elif hasattr(image, 'getvalue'):
        # If image is a Streamlit UploadedFile or camera input
        file_bytes = np.asarray(bytearray(image.getvalue()), dtype=np.uint8)
        img_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        # Also create PIL image from bytes for color detection
        img_pil = Image.open(image)
        img_pil = ImageOps.exif_transpose(img_pil)
        img_pil = img_pil.convert("RGB")
    else:
        # Assume it's already a numpy array
        img_cv2 = image
        # Convert to PIL for color detection
        img_pil = Image.fromarray(img_cv2)
    
    # Run detection
    results = model(img_cv2, verbose=False)
    
    # Convert to supervision Detections format
    detections = sv.Detections.from_ultralytics(results[0]).with_nms()

    # Create custom labels list
    custom_labels = []
    color_counts = {}
    
    for i, box in enumerate(detections.xyxy):
        # Detect color
        avg_color = detect_color_from_bbox(img_pil, box)
        color = identify_color(*avg_color)
        
        # Count colors
        color_counts[color] = color_counts.get(color, 0) + 1
        
        # Create custom label
        if hasattr(detections, 'class_id') and i < len(detections.class_id):
            class_id = detections.class_id[i]
            class_name = model.names[class_id]
            if hasattr(detections, 'confidence') and i < len(detections.confidence):
                conf = detections.confidence[i]
                label = f"{class_name} ({color}, {conf:.2f})"
            else:
                label = f"{class_name} ({color})"
        else:
            label = f"Object ({color})"
            
        custom_labels.append(label)

    # Annotate image with bounding boxes
    box_annotator = sv.BoxAnnotator(thickness=5, color=sv.Color.GREEN)
    annotated_image = img_cv2.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    
    # Use custom labels for annotation
    if custom_labels:
        label_annotator = sv.LabelAnnotator()
        annotated_image = label_annotator.annotate(
            scene=annotated_image, 
            detections=detections,
            labels=custom_labels
        )

    return annotated_image, color_counts