import streamlit as st
import torch
from train import train_image
from torchvision.transforms import ToPILImage
from model import style_transfer_VGG
import time

st.set_option('server.maxUploadSize', 3600)
# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = style_transfer_VGG().to(device).eval()

def main():
    st.title("Style Transfer App")

    # Upload the content image and style image
    content_image = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
    style_image = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

    if content_image and style_image:
        # Show button to start style transfer
        if st.button("Generate Styled Image"):
            with st.spinner("Generating styled image..."):
                # Display placeholder texts while model is running
                text_placeholder = st.empty()
                placeholders = [
                "Let me just rinse my brush...",
                "Carefully outlining the borders...",
                "Ah... this looks very nice...",
                "I like the choice of images...",
                "Adding the final touches...",
                "Mixing colors on my palette...",
                "Creating depth with shading...",
                "Enhancing the highlights...",
                "Working on the textures...",
                "Layering the brushstrokes...",
                "Balancing the composition...",
                "Capturing the essence...",
                "Playing with light and shadow...",
                "Focusing on the finer details...",
                "Stepping back to evaluate...",
                "Adjusting the color balance...",
                "Adding a touch of magic...",
                "Conveying emotion through art...",
                "Feeling the creative flow...",
                "Exploring new horizons...",
                "Crafting a masterpiece...",
                "Bringing imagination to life...",
                "Transforming pixels into art...",
                "Unlocking creativity...",
                "Expressing thoughts through art...",
                "Celebrating the beauty of creation...",
                "Transcending reality...",
                "Embracing the unknown...",
                "Gosh, this is embarrassing...",
                "Please don't leave the app, I promise I'm doing my best!!",
                "You know what, this might be your network...",
                "If the image turns out nice, would you recommend me as an artist?",
                "Gosh, I'm a horrid painter...",
                "Well, at least I'm not spilling paint all over the floor...",
                "Oops, that wasn't supposed to look like that...",
                "Artistic genius or a happy little accident?",
                "I'm painting with pixels, no brushes needed!",
                "Art is never finished, only abandoned...",
                "Painting the canvas of your imagination...",
                "Breaking the rules and creating magic...",
                "Art is like life, always evolving...",
                "Coloring outside the lines to find the magic...",
                "Brace yourself, artistic masterpiece incoming...",
                "Hold onto your seats, pixels are about to transform...",
                "It's not a bug, it's a creative feature!"
            ]

                for text in placeholders:
                    text_placeholder.text(text)
                    time.sleep(4)
                # Perform style transfer and get generated image
                generated_image = train_image(content_image, style_image)

                # Convert the tensor to a PIL Image
                to_pil = ToPILImage()
                generated_pil = to_pil(generated_image.cpu().squeeze().detach())

                # Display the generated image in the app
                st.image(generated_pil, caption="Generated Image", use_column_width=True, channels="RGB")

if __name__ == "__main__":
    main()
