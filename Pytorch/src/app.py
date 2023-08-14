import streamlit as st
import subprocess
from PIL import Image
from train import train_image

st.title("Style Transfer App")

# Upload the content image and style image
content_image = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_image = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

if content_image and style_image:
    # Save uploaded images to disk
    content_path = "content_image.jpg"
    style_path = "style_image.jpg"
    with open(content_path, "wb") as f:
        f.write(content_image.read())
    with open(style_path, "wb") as f:
        f.write(style_image.read())
    
    # Call train_image function to generate the output image
    train_image(content_path, style_path)
    
    # Display the generated image
    generated_img = Image.open("generated_output.png")
    st.image(generated_img, caption="Generated Image", use_column_width=True)





