import numpy as np
import streamlit as st
from train import train_image
import time

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

        if st.button("Generate Styled Image"):
            # Display dynamic text
            text_placeholder = st.empty()
            for text in ["Studying your original photo", "Studying your style photo", "Oops, made a mistake, gotta erase that", "Almost got it"]:
                text_placeholder.text(text)
                time.sleep(7)

            # Perform style transfer and get generated image
            generated_image = train_image(content_path, style_path)
            
            # Convert the tensor to a NumPy array and normalize it
            generated_np = generated_image.squeeze().cpu().detach().numpy()
            generated_np = (generated_np - generated_np.min()) / (generated_np.max() - generated_np.min())
            
            # Display the generated image in the app
            st.image(generated_np, caption="Generated Image", use_column_width=True, channels="RGB")

if __name__ == "__main__":
    main()







