import streamlit as st
import time
from train import train_image

st.title("Style Transfer App")

# Upload the content image and style image
content_image = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_image = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])



# Placeholder to display progress text
text_placeholder = st.empty()

# Placeholder for the generated image
image_placeholder = st.empty()

# Button to start style transfer
if st.button("Generate Styled Image"):
    # Placeholder texts simulating artist's comments
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

    # Perform style transfer and get generated image
    generated_image = train_image(content_image, style_image)

    # Convert the tensor to a NumPy array and normalize it
    generated_np = generated_image.squeeze().cpu().detach().numpy()
    generated_np = (generated_np - generated_np.min()) / (generated_np.max() - generated_np.min())

    # Display progress text using placeholders
    for text in placeholders:
        text_placeholder.text(text + "...")
        time.sleep(7)
        if text == "Adding the final touches...":
            # Display the generated image in the app
            # Display uploaded image
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(generated_image, caption="Uploaded Image", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            text_placeholder.empty()  # Remove the progress text
            break  # Stop displaying texts after image is generated







