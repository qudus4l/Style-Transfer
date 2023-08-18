# Artistic Style Transfer for Images and Videos

Welcome to our repository showcasing Python scripts that enable the application of artistic style transfer to both real-time webcam videos and pre-recorded videos/images. This implementation harnesses the power of TensorFlow, PyTorch, and OpenCV to create captivating visual transformations.

## Example Stylized Image

Behold a captivating stylized image produced by our script:

![Stylized Image Example](https://github.com/qudus4l/Style-Transfer/assets/110972011/7330d8cc-e9b8-4e1c-a5ff-a226c9bff096)

## Example Stylized GIF

Observe the mesmerizing GIF stylization in action:

![Stylized GIF Example](https://github.com/qudus4l/Style-Transfer/assets/110972011/e2b706e3-6664-46ab-8633-42442a29a98e)

*NOTE:* Our solution isn't limited to images and GIFs – it's also applicable to full-fledged videos in formats such as MP4 and MPEG4. For real-time stylization, you'll need to run the `streamlit_app.py` script on your localhost.

## Getting Started

Follow these steps to set up and run the scripts on your local machine.

### Prerequisites

Before you begin, ensure that you have the following software installed:

- Python (version 3.6 or higher)
- Git

### Cloning the Repository

To acquire the repository on your local machine, open a terminal and execute the following command:

```sh
git clone https://github.com/qudus4l/Style-Transfer.git
```

### Installing Dependencies

Navigate to the project directory:

```sh
cd Style-Transfer
```

Install the necessary Python packages using the `pip` package manager:

```sh
pip install -r requirements.txt
```

### Running the Script

To initiate the real-time style transfer script using Streamlit, execute the following command:

```sh
streamlit run streamlit_app.py
```

### Stopping the Script

During script execution, you can halt the real-time style transfer by closing the Streamlit app.

## Acknowledgments

Our implementation leverages a variety of technologies, including TensorFlow, TensorFlow Hub, PyTorch, and other libraries, to achieve real-time style transfer.

## License

This project operates under the MIT License. Please refer to the [LICENSE](LICENSE) file for detailed information.
