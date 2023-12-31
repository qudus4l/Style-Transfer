import streamlit as st
import random
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from PIL import Image

@st.cache(suppress_st_warning=True)
def preprocess_and_view_image(content_image_path, style_image_path):
    content_image = plt.imread(content_image_path)
    style_image = plt.imread(style_image_path)
    # Scale the images to ensure they are within the range of [0, 1]
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    # Resize the style image
    style_image = tf.image.resize(style_image, [256, 256])
    # Load the module
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]
    # Convert the tensor to a NumPy array
    stylized_image_array = stylized_image.numpy()

    # Ensure values are within the valid range [0, 1]
    stylized_image_array = np.clip(stylized_image_array, 0, 1)

    # Close the matplotlib figure to prevent display in the notebook
    plt.close()

    return stylized_image_array[0]

@st.cache(suppress_st_warning=True)
def preprocess_and_display_video(video_path, style_image_path):
    '''This function preprocesses the video and applies style transfer'''

    # Load our style transfer module
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Capture the video file and transform it into a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Preprocess the style image
    style_image = plt.imread(style_image_path)
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, [256, 256])

    stylized_frames = []  # List to store stylized frames

    # Create a while loop to read and stylize each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess the video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)[np.newaxis, ...] / 255
        frame = tf.image.resize(frame, [256, 256])

        # Apply stylization
        stylized_output = hub_module(tf.constant(frame), tf.constant(style_image, dtype=tf.float32))
        stylized_frame = (stylized_output[0].numpy()[0] * 255).astype(np.uint8)
        stylized_frame = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)

        stylized_frames.append(stylized_frame)

    # Release video object
    cap.release()

    return stylized_frames


# Function for the "Time Travel" feature
def time_travel_app():
    st.title("TimeFlow: Artistic Odyssey")

    st.markdown(
    """
    <style>
        /* Set the font to Comic Sans MS */
        body {
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
    </style>

    **Welcome to ArtVoyage: Temporal Expressions!** 🎨

    Embark on a journey through time and artistry, where you can explore the intersection
    of historical epochs and artistic styles. Choose an era to discover captivating
    masterpieces from renowned artists of that period.

    Begin your artistic adventure now! 🚀
    """
    , unsafe_allow_html=True
)
    selected_era = st.selectbox("🕰️ **Choose an era that captivates you!**", [
    "Ancient Art",
    "Medieval Art",
    "Renaissance Art",
    "Mannerism Art",
    "Baroque Art",
    "Rococo Art",
    "Neoclassicism Art",
    "Romanticism Art",
    "Realism Art",
    "Art Nouveau Art",
    "Impressionism Art",
    "Post-Impressionism Art",
    "Fauvism Art",
    "Expressionism Art",
    "Contemporary Art"
])

    artists_by_era = {
    "Ancient Art": [
        "Bularchus", "Panaenus"],
    "Medieval Art": [
        "Giotto di Bondone","Hildegard of Bingen","Limbourg Brothers","Cimabue","Master Bertram"],
    "Renaissance Art": [
        "Leonardo da Vinci","Michelangelo Buonarroti","Raphael","Sandro Botticelli","Albrecht Dürer"
    ],
    "Mannerism Art": [
        "El Greco","Parmigianino","Jacopo da Pontormo","Rosso Fiorentino","Giuseppe Arcimboldo"
    ],
    "Baroque Art": [
        "Caravaggio","Rembrandt van Rijn","Peter Paul Rubens"
    ],
    "Rococo Art": [
        "Jean-Antoine Watteau","François Boucher","Jean-Honoré Fragonard","Thomas Gainsborough","Jean-Baptiste-Siméon Chardin"
    ],
    "Neoclassicism Art": [
        "Jacques-Louis David","Jean-Auguste-Dominique Ingres","Angelica Kauffman"
    ],
    "Romanticism Art": [
        "Caspar David Friedrich","Eugène Delacroix","J.M.W. Turner","Francisco Goya","William Blake"
    ],
    "Realism Art": [
        "Honoré Daumier","Jean-François Millet","Winslow Homer","Thomas Eakins"
    ],
    "Art Nouveau Art": [
        "Alphonse Mucha","Antoni Gaudí","Aubrey Beardsley","Gustav Klimt"
    ],
    "Impressionism Art": [
        "Claude Monet","Pierre-Auguste Renoir","Camille Pissarro","Berthe Morisot"
    ],
    "Post-Impressionism Art": [
        "Vincent van Gogh","Paul Cézanne","Georges Seurat","Paul Gauguin","Henri Rousseau"
    ],
    "Fauvism Art": [
        "Henri Matisse","André Derain","Raoul Dufy","Kees van Dongen","Albert Marquet"
    ],
    "Expressionism Art": [
        "Edvard Munch","Ernst Ludwig Kirchner","Wassily Kandinsky","Egon Schiele","Emil Nolde"
    ],
    "Contemporary Art": [
        "Damien Hirst","Jeff Koons","Ai Weiwei","Yayoi Kusama","Anti Shade"
    ]
}



    selected_artist = random.choice(artists_by_era[selected_era])

    artist_folder = os.path.join("artists", selected_artist)
    artist_artworks = os.listdir(artist_folder)
    selected_artwork_path = os.path.join(artist_folder, random.choice(artist_artworks))

    user_image = st.file_uploader("🖼️🧑‍🎨Let's see how you'd look if you were painted by an artist from that era!", type=["jpg", "jpeg", "png"])

    if user_image:

        if st.button("Generate Stylized Image"):
            with st.spinner("Generating styled image..."):
                    text_placeholder = st.empty()
                    placeholders = [
                        "Alright! Going back in time!",
                        "Locating the artist...",
                        "Painting...",
                        "Adding the final touches...",
                        "Stepping back to evaluate...",
                        "Adding a signature...",
                        "All done! Here's your art!"
                    ]

                    for text in placeholders:
                        text_placeholder.text(text)
                        time.sleep(2)
            stylized_image = preprocess_and_view_image(user_image, selected_artwork_path)
            st.image(stylized_image, caption=f"Here's what you would look like if you were painted by {selected_artist}", width=400)


# Function for the "Neural Style Transfer" feature
def neural_style_transfer_app():
    st.title("Neural Style Transfer")
    st.markdown(
    """
    <style>
        /* Set the font to Comic Sans MS */
        body {
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
    </style>


    Elevate your visuals with the Neural Style Transfer App! 
    Combine your images and videos with artistic styles to create captivating, 
    unique content. Choose from:

    Begin your artistic adventure now! 🚀
    """
    , unsafe_allow_html=True
)

    content_type = st.radio("Choose Input Type", ["Image", "Video", "Real-time"])

    if content_type == "Image":
        st.write("Transform your images using artistic styles.")
        content_image_path = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
        style_image_path = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

        if content_image_path and style_image_path:
            if st.button("Generate Styled Image"):
                with st.spinner("Generating styled image..."):
                    text_placeholder = st.empty()
                placeholders = [
                    "Let me just rinse my brush...",
                    "Carefully outlining the borders...",
                    "Ah... this looks very nice...",
                    "I like the choice of images...",
                    "Adding the final touches...",
                    "Stepping back to evaluate...",
                    "Adding a signature...",
                    "All done! Here's your image!"
                ]

                for text in placeholders:
                    text_placeholder.text(text)
                    time.sleep(2)
                stylized_image = preprocess_and_view_image(content_image_path, style_image_path)
                st.image(stylized_image, caption="Stylized Image", width=400)

    elif content_type == "Video":
        st.write("Give your videos an artistic touch by applying captivating styles.")
        content_video_path = st.file_uploader("Upload Content Video", type=["mp4", "gif"])
        style_image_path = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

        if content_video_path and style_image_path:
            if st.button("Generate Styled Video"):
                with st.spinner("Generating styled video..."):
                    with open("temp_video.mp4", "wb") as temp_file:
                        temp_file.write(content_video_path.read())
                        text_placeholder = st.empty()
                    placeholders = [
                        "Nice video...",
                        "Splitting the frames",
                        "Give me a sec...",
                        "I like the choice of style image...",
                        "Adding the final touches...",
                        "Combining the frames",
                        "Adding a signature...",
                        "All done! Here's your video!"
                    ]

                    for text in placeholders:
                        text_placeholder.text(text)
                        time.sleep(2)
                    stylized_frames = preprocess_and_display_video("temp_video.mp4", style_image_path)
    
                    # Save stylized frames as a looping GIF using PIL
                    gif_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in stylized_frames]
                    gif_frames[0].save("stylized_video.gif", save_all=True, append_images=gif_frames[1:], loop=0, duration=100)

                    # Display the looping GIF
                    gif_bytes = open("stylized_video.gif", "rb").read()
                    st.image(gif_bytes)

                    # Clean up the temporary files
                    if os.path.exists("temp_video.mp4"):
                        os.remove("temp_video.mp4")
                    if os.path.exists("stylized_video.gif"):
                        os.remove("stylized_video.gif")
                    
    elif content_type == "Real-time":
        st.write("Experience live style transformation using your webcam – see the magic happen as you move (Run on local).")
        style_image_path = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
        if style_image_path:
            # Load the style transfer model
            hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

            # Add a button to start stylization
            start_button = st.button("Start Stylization")

            if start_button:
                # Capture video from webcam
                org_video = cv2.VideoCapture(0)

                # Get frame dimensions
                frame_width = int(org_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(org_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(org_video.get(cv2.CAP_PROP_FPS))

                # Preprocess the style image
                style_image = plt.imread(style_image_path)
                style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
                style_image = tf.image.resize(style_image, [256, 256])

                # Create VideoWriter object
                out = cv2.VideoWriter('stylized_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

                # Add a stop button
                stop_button = st.button("Stop Stylization")

                # Create a placeholder for displaying video frames
                frame_placeholder = st.empty()

                # Loop for real-time video stylization
                while org_video.isOpened() and not stop_button:
                    ret, frame = org_video.read()
                    if not ret:
                        break

                    # Preprocess the frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32)[np.newaxis, ...] / 255
                    frame = tf.image.resize(frame, [256, 256])

                    # Apply stylization
                    stylized_output = hub_module(tf.constant(frame), tf.constant(style_image, dtype=tf.float32))
                    stylized_video_output = stylized_output[0].numpy()[0]

                    # Postprocess the video
                    stylized_video_output = (stylized_video_output * 255).astype(np.uint8)
                    stylized_video_output = cv2.cvtColor(stylized_video_output, cv2.COLOR_RGB2BGR)

                    out.write(stylized_video_output)

                    # Display the stylized video in the placeholder
                    frame_placeholder.image(stylized_video_output, channels="RGB", use_column_width=True)

                    # Introduce a small delay to simulate real-time camera experience
                    time.sleep(0.1)

                # Release video and writer
                org_video.release()
                out.release()

# Define the main navigation sidebar
st.sidebar.title("What would you like to do")
page_names_to_funcs = {
    "Travel Back in time": time_travel_app,
    "Neural Style Transfer": neural_style_transfer_app,
}
demo_name = st.sidebar.selectbox("Pick a feature", list(page_names_to_funcs.keys()))
page_names_to_funcs[demo_name]()

# Footer
st.markdown('<div class="footer">Created by <a href="https://www.linkedin.com/in/qudus-abolade/">Qudus</a> & <a href="https://www.linkedin.com/in/oluwafikayomi-adeleke-98a29023b">Fikayo</a>🦇</div>', unsafe_allow_html=True)
