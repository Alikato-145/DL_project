import streamlit as st
import os
import cv2
import torch
from PIL import Image, ImageOps

# Import the patched ultralytics first
from fix_ultralytics import ultralytics
from ultralytics import YOLO
import supervision as sv
from detection import detection

# Set the page configuration
st.set_page_config(
    page_title="Plate Detection and Cost Calculation",
    page_icon=":guardsman:",
    initial_sidebar_state="expanded",
)

# using cpu for inference
device = torch.device("cpu")
print(ultralytics.__version__)
print(torch.__version__)
# Title for the app
st.title("Plate Detection and Cost Calculation")
st.subheader("dev : ข้าวทุกจาน อาหารทุกอย่าง team")

# Initialize YOLO model with loaded weights
@st.cache_resource  # Cache the model to avoid reloading it every time
def load_model():
    return YOLO("project/lts_model/runs/detect/train/best.pt")
model = load_model()  # To check if the model loads correctly

# Add price configuration in sidebar
st.sidebar.header("Price Configuration")
prices = {}
colors = ["Red", "Yellow", "Green", "Blue", "Cyan", "Purple", "White", "Black"]
for color in colors:
    prices[color] = st.sidebar.number_input(f"{color} Plate Price", min_value=0, value=0, step=5)

# choose to use camera or upload files
st.subheader("Choose to use camera or upload files")
camera = st.camera_input("Take a picture")
uploadfiles = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "JPG"], accept_multiple_files=True)

# Check if the camera input is used
if camera:
    # Create a directory to save the images
    if not os.path.exists("images"):
        os.makedirs("images")

    # Save the image from the camera input
    file_path = os.path.join("images", "captured_image.jpg")
    with open(file_path, "wb") as f:
        f.write(camera.getbuffer())

    st.success("Image saved successfully!")

    # Display the captured image
    st.image(camera, caption="Captured Image", use_container_width=True)

    # Run detection on the captured image
    annotated_image, color_counts = detection(file_path, model)
    
    # Save the annotated image
    annotated_image_path = os.path.join("images", "annotated_captured_image.jpg")
    cv2.imwrite(annotated_image_path, annotated_image)

    # Display the annotated image
    st.image(annotated_image, caption="Annotated Image", use_container_width=True)
    
    # Display color counts and calculate total price
    st.subheader("Color Counts")
    total_price = 0
    for color, count in color_counts.items():
        price = prices.get(color, 0)
        item_total = price * count
        total_price += item_total
        st.write(f"{color}: {count} plates × {price} = {item_total}")
    
    st.subheader(f"Total Price: {total_price}")

if uploadfiles:
    # Create a directory to save the images
    if not os.path.exists("images"):
        os.makedirs("images")

    # Save the images to the directory
    for file in uploadfiles:
        file_path = os.path.join("images", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.success("Images saved successfully!")

    # Process each uploaded image
    for file in uploadfiles:
        st.write(f"## Processing: {file.name}")
        
        # Display the uploaded image
        st.image(file.getvalue(), caption=file.name, use_container_width=True)
        
        # Run detection for the uploaded image
        file_path = os.path.join("images", file.name)
        annotated_image, color_counts = detection(file_path, model)

        # Save the annotated image
        annotated_image_path = os.path.join("images", "annotated_" + file.name)
        cv2.imwrite(annotated_image_path, annotated_image)

        # Display the annotated image
        st.image(annotated_image, caption="Annotated Image", use_container_width=True)
        
        # Display color counts and calculate total price
        st.subheader("Color Counts")
        total_price = 0
        for color, count in color_counts.items():
            price = prices.get(color, 0)
            item_total = price * count
            total_price += item_total
            st.write(f"{color}: {count} plates × {price} = {item_total}")
        
        st.subheader(f"Total Price: {total_price}")
        st.markdown("---")