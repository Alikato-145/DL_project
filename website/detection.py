import cv2
import numpy as np
from PIL import Image, ImageOps
import supervision as sv

def histogram_equalization(image):
    """
    ทำ Histogram Equalization บนช่อง Value (V) ของภาพ HSV
    """
    # แปลงจาก RGB เป็น HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # แยกช่อง Hue, Saturation, Value
    h, s, v = cv2.split(hsv)

    # ทำ Histogram Equalization บนช่อง V (ความสว่าง)
    v_eq = cv2.equalizeHist(v)

    # รวมกลับเป็น HSV
    hsv_eq = cv2.merge([h, s, v_eq])
    # แปลงกลับเป็น RGB
    image_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    return image_eq

def white_balance_simple(image):
    """
    Simple white balance by scaling channels based on average intensity
    """
    # Convert to float for calculations
    image_float = image.astype(np.float32)

    # Calculate the mean for each channel
    mean_r = np.mean(image_float[:,:,0])
    mean_g = np.mean(image_float[:,:,1])
    mean_b = np.mean(image_float[:,:,2])

    # Calculate scaling factors
    scale_r = 128 / mean_r if mean_r > 0 else 1
    scale_g = 128 / mean_g if mean_g > 0 else 1
    scale_b = 128 / mean_b if mean_b > 0 else 1

    # Scale each channel
    image_float[:,:,0] *= scale_r
    image_float[:,:,1] *= scale_g
    image_float[:,:,2] *= scale_b

    # Clip values to valid range
    image_balanced = np.clip(image_float, 0, 255).astype(np.uint8)

    return image_balanced

def white_balance_gray_world(image):
    """
    Gray World Assumption white balance method
    """
    # Convert to float for calculations
    image_float = image.astype(np.float32)

    # Calculate average of each channel
    r_avg = np.mean(image_float[:,:,0])
    g_avg = np.mean(image_float[:,:,1])
    b_avg = np.mean(image_float[:,:,2])

    # Calculate the overall average
    avg = (r_avg + g_avg + b_avg) / 3

    # Calculate scaling factors
    r_scale = avg / r_avg if r_avg > 0 else 1
    g_scale = avg / g_avg if g_avg > 0 else 1
    b_scale = avg / b_avg if b_avg > 0 else 1

    # Apply scaling
    image_float[:,:,0] *= r_scale
    image_float[:,:,1] *= g_scale
    image_float[:,:,2] *= b_scale

    # Clip values to valid range
    image_balanced = np.clip(image_float, 0, 255).astype(np.uint8)
    return image_balanced

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

    # Print debug information
    print(f"🔍 Hue: {hue:.2f}, Saturation: {sat:.2f}%, Value: {val:.2f}%")

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
        return "Red"

    return "Unknown"

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
    - box1, box2: [x1, y1, x2, y2] format bounding boxes
    
    Returns:
    - IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0  # No intersection
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def filter_duplicate_detections(detections, iou_threshold=0.5, area_ratio_threshold=0.8):
    """
    Filter out duplicate detections based on IoU and area ratio
    
    Args:
    - detections: sv.Detections object
    - iou_threshold: Threshold for IoU to consider boxes as duplicates
    - area_ratio_threshold: Threshold for area ratio to prefer larger boxes
    
    Returns:
    - Filtered sv.Detections object
    """
    if len(detections) <= 1:
        return detections
    
    # Get detections data
    boxes = detections.xyxy
    confidences = detections.confidence
    
    # Calculate areas for each box
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    
    # Create a list of detection indices sorted by confidence
    indices = list(range(len(boxes)))
    indices.sort(key=lambda i: confidences[i], reverse=True)
    
    keep_indices = []
    
    # Process boxes in order of confidence
    for i in range(len(indices)):
        current_idx = indices[i]
        
        # If already processed and removed, skip
        if current_idx not in indices:
            continue
        
        keep = True
        current_box = boxes[current_idx]
        current_area = areas[current_idx]
        
        # Compare with all previously kept boxes
        for kept_idx in keep_indices:
            kept_box = boxes[kept_idx]
            kept_area = areas[kept_idx]
            
            # Calculate IoU
            iou = calculate_iou(current_box, kept_box)
            
            # Calculate area ratio (smaller/larger)
            area_ratio = min(current_area, kept_area) / max(current_area, kept_area)
            
            # If boxes overlap significantly and have similar sizes, consider as duplicate
            if iou > iou_threshold and area_ratio > area_ratio_threshold:
                keep = False
                break
        
        if keep:
            keep_indices.append(current_idx)
    
    # Create a new Detections object with only the kept indices
    if keep_indices:
        return detections[keep_indices]
    else:
        return detections[0:0]  # Empty detections

def detection(file_path,model,conf_threshold=0.4, iou_threshold=0.5):
    """
    Main detection function that matches the process_image function from your notebook
    
    Args:
    - file_path: Path to the image file
    - model: Loaded object detection model
    
    Returns:
    - annotated_image: PIL Image with annotations
    - color_counts: Dictionary with color counts
    """
    # Use same implementation as process_image from notebook
    image = Image.open(file_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    #image = Image.fromarray(histogram_equalization(np.array(image)))

    # Detect objects - exactly as in the notebook
    results = model(image, verbose=False, conf=conf_threshold)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()
    filtered_detections = filter_duplicate_detections(
        detections, 
        iou_threshold=iou_threshold,
        area_ratio_threshold=0.5
    )
    # Annotate image
    box_annotator = sv.BoxAnnotator(thickness=5)
    label_annotator = sv.LabelAnnotator()
    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Count colors - same as notebook
    dict_count = {}
    for i, box in enumerate(detections.xyxy):
        avg_color = detect_color_from_bbox(image, box)
        color = identify_color(*avg_color)

        # Count colors
        dict_count[color] = dict_count.get(color, 0) + 1

    # Print results - keep identical to notebook
    for key, value in dict_count.items():
        print(f"{key}: {value}")

    print(f"Number of detections: {len(detections.xyxy)}")
    print(f"Number of boxes drawn: {len(detections)}")

    # Return exactly what's needed
    return annotated_image, dict_count