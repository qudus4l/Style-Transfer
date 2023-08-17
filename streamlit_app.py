import streamlit as st
import random
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

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
    Unveil the hidden connections between the past and present by applying neural style
    transfer to your own images using iconic artworks as inspiration. Get ready to create
    a fusion of timeless art and contemporary vision.

    Begin your artistic adventure now! 🚀
    """
    , unsafe_allow_html=True
)


    selected_era = st.selectbox("🕰️ **Choose an era that captivates you!**", ["Ancient Art", "Medieval Art", "Renaissance"])

    artists_by_era = {
        "Ancient Art": ["Bularchus", "Panaenus"],
        "Medieval Art": ["Giotto di Bondone", "Hildegard of Bingen", "Limbourg Brothers", "Cimabue", "Master Bertram"],
        "Renaissance": ["Leonardo da Vinci", "Michelangelo Buonarroti", "Raphael", "Sandro Botticelli"]
    }

    selected_artist = random.choice(artists_by_era[selected_era])

    artist_folder = os.path.join("artists", selected_artist)
    artist_artworks = os.listdir(artist_folder)
    selected_artwork_path = os.path.join(artist_folder, random.choice(artist_artworks))

    user_image = st.file_uploader("🖼️🧑‍🎨Let's see how you'd look if you were painted by an artist from that era!", type=["jpg", "jpeg", "png"])

    if user_image:

        if st.button("Generate Stylized Image"):
            stylized_image = preprocess_and_view_image(user_image, selected_artwork_path)
            st.image(stylized_image, caption=f"Here's what you would look like if you were painted by {selected_artist}", width=400)



# Function for the "Neural Style Transfer" feature
def neural_style_transfer_app():
    st.title("Neural Style Transfer")

    content_type = st.radio("Choose Input Type", ["Image", "Video"])

    if content_type == "Image":
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
        content_video_path = st.file_uploader("Upload Content Video", type=["mp4", "gif"])
        style_image_path = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

        if content_video_path and style_image_path:
            if st.button("Generate Styled Video"):
                with st.spinner("Generating styled video..."):
                    with open("temp_video.mp4", "wb") as temp_file:
                        temp_file.write(content_video_path.read())
                    stylized_frames = preprocess_and_display_video("temp_video.mp4", style_image_path)
    
                    # Create a video from stylized frames using OpenCV's VideoWriter
                    frame_height, frame_width, _ = stylized_frames[0].shape

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter("stylized_video.mp4", fourcc, 30, (frame_width, frame_height))

                    for frame in stylized_frames:
                        out.write(frame)
                    out.release()

                    # Read the stylized video back and display it using Streamlit
                    stylized_video = open("stylized_video.mp4", "rb").read()
                    st.video(stylized_video)

                    # Clean up the temporary files
                    if os.path.exists("temp_video.mp4"):
                        os.remove("temp_video.mp4")
                    if os.path.exists("stylized_video.mp4"):
                        os.remove("stylized_video.mp4")



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
