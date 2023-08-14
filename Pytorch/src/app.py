import streamlit as st
import torch
from train import train_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
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
        
        # Perform style transfer and get generated image
        generated_image = train_image(content_path, style_path)
        
        # Display the generated image in the app
        st.image(generated_image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()






