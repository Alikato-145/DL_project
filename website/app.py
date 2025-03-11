import streamlit as st
import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import time

from mae import calculate_mae

# Import the patched ultralytics first
from fix_ultralytics import ultralytics
from ultralytics import YOLO
import supervision as sv
from detection import detection

# Set page configuration with wider layout and custom theme
st.set_page_config(
    page_title="Plate Detection and Cost Calculation",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .section-header {
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    .price-card {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight-total {
        font-size: 2rem;
        color: #1E88E5;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .stTable {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }
    .upload-section, .camera-section {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .color-dot {
        height: 15px;
        width: 15px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .progress-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        margin: 20px 0;
    }
    .result-section {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header section with logo and title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🍽️ Plate Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ข้าวทุกจาน อาหารทุกอย่าง team</p>', unsafe_allow_html=True)

# Initialize device and model
device = torch.device("cpu")

# Load model with a proper loading indicator
with st.spinner("Loading detection model..."):
    @st.cache_resource
    def load_model():
        return YOLO("project/model_demo/best.pt")
    
    try:
        model = load_model()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Create two tabs for different input methods
tab1, tab2 = st.tabs(["📷 Camera Input", "📁 Upload Images"])

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Price Configuration</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="price-card">', unsafe_allow_html=True)
    
    # Add color indicators to price inputs

    prices = {}
    colors = ["Red", "Yellow", "Green", "Blue", "Cyan", "Purple", "White", "Black"]
    colors_thai = ["ราคาจานสีแดง", "ราคาจานสีเหลือง", "ราคาจานสีเขียว", "ราคาจานสีน้ำเงิน", "ราคาจานสีเขียวแกมน้ำเงิน", "ราคาจานสีม่วง", "ราคาจานสีขาว", "ราคาจานสีดำ"]
    color_hex = {
        "Red": "#FF5252", 
        "Yellow": "#FFD740", 
        "Green": "#69F0AE", 
        "Blue": "#448AFF", 
        "Cyan": "#18FFFF", 
        "Purple": "#E040FB", 
        "White": "#FFFFFF", 
        "Black": "#212121"
    }

    for color, color_th in zip(colors, colors_thai):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f'<div style="display: flex; align-items: center; height: 38px;"><span class="color-dot" style="background-color: {color_hex[color]}; width: 20px; height: 20px; border-radius: 50%; display: inline-block;"></span></div>', unsafe_allow_html=True)
        with col2:
            prices[color] = st.number_input(color_th, min_value=0, value=0, step=5)

    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🛠️ Advanced Settings</div>', unsafe_allow_html=True)
    conf_threshold = 0.3
    iou_threshold = 0.5
    
    st.markdown('<div class="section-header">ℹ️ App Info</div>', unsafe_allow_html=True)
    st.info("""
    This application detects colored plates in images and calculates the total cost based on your price configuration.
    
    **Usage:**
    1. Set prices for each plate color
    2. Take a photo or upload images
    3. View detection results and cost calculation
    """)

# Function to create and display a cost table
def display_cost_table(color_counts, prices):
    # Create data for the table
    table_data = []
    total_price = 0
    
    # Sort colors by count (descending)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    
    for color, count in sorted_colors:
        price = prices.get(color, 0)
        item_total = price * count
        total_price += item_total
        
        # Add color indicator
        color_indicator = f'<span class="color-dot" style="background-color: {color_hex.get(color, "#CCCCCC")};"></span> {color}'
        
        table_data.append({
            "Color": color_indicator,
            "Count": count,
            "Price per Plate": f"฿{price}",
            "Subtotal": f"฿{item_total}"
        })
    
    # Create a DataFrame for the table
    df = pd.DataFrame(table_data)
    
    # Display the table with HTML formatting for color dots
    st.write("### Detailed Cost Breakdown")
    st.markdown(pd.DataFrame(df).to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Display a large, highlighted total
    st.markdown(f'<div class="highlight-total">💰 Total Cost: ฿{total_price}</div>', unsafe_allow_html=True)
    
    return 

def calculate_mae_and_display(real_counts, color_counts):
    # Calculate MAE
    mae_result = calculate_mae(real_counts, color_counts)
    
    # Display MAE results
    st.markdown('<div class="section-header">📊 Evaluation Results</div>', unsafe_allow_html=True)
    
    st.write("### Mean Absolute Error (MAE) Results")
    st.write(f"**Total Plates Detected:** {mae_result['total_predicted']}")
    st.write(f"**Total Plates Real:** {mae_result['total_real']}")
    st.write(f"**Absolute Total Error:** {mae_result['absolute_total_error']}")
    
    st.write(f"**Mean Absolute Error (MAE):** {mae_result['mae']}")
    st.write("### Detailed Color Errors")
    for color, error in mae_result['color_errors'].items():
        st.write(f"- {color}: {error} plates")
    st.write(f"**Total Error:** {mae_result['total_error']} plates")


# Function to process images and display results
def process_image(file_path, filename=None):
    with st.spinner("Detecting plates..."):
        # Show progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            # Simulate processing time
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Run detection with advanced parameters
        try:
            annotated_image, color_counts = detection(
                file_path, 
                model, 
                conf_threshold=conf_threshold, 
                iou_threshold=iou_threshold
            )
            
            # Save the annotated image
            if not os.path.exists("images"):
                os.makedirs("images")
            
            # FIX: Handle different image formats appropriately
            annotated_image_path = os.path.join("images", f"annotated_{filename or 'captured_image.jpg'}")
            
            # Check if annotated_image is a PIL Image and convert if needed
            if hasattr(annotated_image, 'convert'):  # Check if it's a PIL Image
                annotated_image = np.array(annotated_image)
                # Convert RGB to BGR if needed (PIL uses RGB, OpenCV uses BGR)
                if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Ensure annotated_image is a valid numpy array
            if annotated_image is not None and isinstance(annotated_image, np.ndarray):
                try:
                    # Attempt to save the image
                    cv2.imwrite(annotated_image_path, annotated_image)
                except Exception as e:
                    st.error(f"Error saving annotated image: {e}")
                    # Fallback: Display the annotated image without saving
                    annotated_image_path = None
            else:
                st.warning("Cannot save annotated image: Invalid image format")
                annotated_image_path = None
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="section-header">📊 Detection Results</div>', unsafe_allow_html=True)
                # Display the image - if we couldn't save it, just show the original
                if annotated_image_path and os.path.exists(annotated_image_path):
                    st.image(annotated_image_path, caption="Detected Plates", use_container_width =True)
                elif isinstance(annotated_image, np.ndarray):
                    # If we have a valid numpy array but couldn't save, convert BGR to RGB for display
                    if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                        display_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    else:
                        display_image = annotated_image
                    st.image(display_image, caption="Detected Plates", use_container_width =True)
                else:
                    # If all else fails, show the original image
                    st.image(file_path, caption="Original Image (Detection Failed)", use_container_width =True)
                
                # Display detection summary
                plate_count = sum(color_counts.values())
                st.metric("Total Plates Detected", plate_count)
                
            with col2:
                st.markdown('<div class="section-header">💵 Cost Calculation</div>', unsafe_allow_html=True)
                display_cost_table(color_counts, prices)
            
            # Store color_counts in session state for MAE calculation
            st.session_state['color_counts'] = color_counts
            
            return color_counts
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return {}

# Camera tab content
with tab1:
    st.markdown('<div class="camera-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📷 Take a Photo</div>', unsafe_allow_html=True)
    
    camera = st.camera_input("", key="camera_input")
    
    if camera:
        # Create a directory to save the images
        if not os.path.exists("images"):
            os.makedirs("images")

        # Save the image from the camera input
        file_path = os.path.join("images", "captured_image.jpg")
        with open(file_path, "wb") as f:
            f.write(camera.getbuffer())

        st.success("Image captured successfully!")
        
        # Process the captured image
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        color_counts = process_image(file_path, "captured_image.jpg")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# Upload tab content
with tab2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📁 Upload Images</div>', unsafe_allow_html=True)
    
    uploadfiles = st.file_uploader(
        "Choose images to upload",
        type=["jpg", "jpeg", "png", "JPG"],
        accept_multiple_files=True,
        help="You can upload multiple images at once"
    )
    
    if uploadfiles:
        # Create a directory to save the images
        if not os.path.exists("images"):
            os.makedirs("images")

        # Save the images to the directory
        for file in uploadfiles:
            file_path = os.path.join("images", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        st.success(f"✅ {len(uploadfiles)} images uploaded successfully!")

        # Create an expander for each uploaded image
        for file in uploadfiles:
            with st.expander(f"📷 {file.name}", expanded=True):
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                
                # Display the uploaded image
                st.image(file.getvalue(), caption=file.name, use_container_width =True)
                
                # Process the uploaded image
                file_path = os.path.join("images", file.name)
                color_counts = process_image(file_path, file.name)
                
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def add_sidebar():
    with st.sidebar:
        st.markdown('<div class="section-header">📊 Evaluation Tool</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="price-card">', unsafe_allow_html=True)
        st.write("Enter the actual plate counts for evaluation:")
        
        # Add input fields for each color
        real_counts = {}
        colors = ["Red", "Yellow", "Green", "Blue", "Cyan", "Purple", "White", "Black"]
        
        for color in colors:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f'<div style="display: flex; align-items: center; height: 38px;"><span class="color-dot" style="background-color: {color_hex[color]}; width: 20px; height: 20px; border-radius: 50%; display: inline-block;"></span></div>', unsafe_allow_html=True)
            with col2:
                real_counts[color] = st.number_input(f"Actual {color} count", min_value=0, value=0, step=1, key=f"real_{color}")
        
        st.session_state['real_counts'] = real_counts
        st.markdown('</div>', unsafe_allow_html=True)

add_sidebar()
# Display MAE results if real counts are provided
if st.sidebar.button("Calculate MAE"):
    real_counts = st.session_state['real_counts']
    color_counts = st.session_state.get('color_counts', {})
    st.markdown('<div class="section-header">📊 MAE Calculation</div>', unsafe_allow_html=True)
    if any(real_counts.values()):
        calculate_mae_and_display(real_counts, color_counts)
    else:
        st.warning("Please enter actual counts to calculate MAE.")
# Display the MAE input section 


# Footer
st.markdown("---")
st.markdown("© 2025 ข้าวทุกจาน อาหารทุกอย่าง team | All Rights Reserved")