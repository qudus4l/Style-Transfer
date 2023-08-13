import streamlit as st
from PIL import Image
from train.py import train_image  # Import the generated image from style_transfer_module.py

def style_transfer_app():
    st.title("Style Transfer App")
    
    try:
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
            
            # Run the style transfer code
            # Replace the generated image with the result of style transfer
            generated = train_image(content_path, style_path)
            
            # Display the generated image
            st.image(generated, caption="Generated Image", use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    style_transfer_app()



