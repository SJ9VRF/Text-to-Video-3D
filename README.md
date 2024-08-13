# Text-to-Video 3D

![Screenshot_2024-08-13_at_12 12 20_AM-removebg-preview](https://github.com/user-attachments/assets/9c0ca0bd-29d4-4542-bfa8-490ded2d04af)


# Text-to-Video Synthesis with VQGAN and CLIP

This project combines the capabilities of VQGAN (Vector Quantized Generative Adversarial Networks) and CLIP (Contrastive Language-Image Pre-training) to generate video frames based on textual descriptions. The synthesis system applies a series of transformations to create dynamic, artistic representations of textual prompts.

## Features

- Text-to-image synthesis using VQGAN+CLIP.
- Image transformations (zoom, rotate, translate) to enhance the visual dynamics.
- Output frames saved as images which can be compiled into videos.

## Prerequisites

- Python 3.8 or higher
- Pip package manager
- Access to a CUDA-compatible GPU for faster processing (optional but recommended).

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repository/Text-to-Video-VQGAN-CLIP.git
   cd Text-to-Video-VQGAN-CLIP
   ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
   ```
3. **Install dependencies**
Download the necessary VQGAN model configuration and checkpoint files. These files should be placed in the appropriate directories specified in the main script.


## Usage
Run the main script to generate frames based on predefined text prompts. Each frame is saved as an image in the output directory.
   ```bash
   python src/main.py
   ```
You can modify the text prompts directly in the main.py file to create different images.

## Extending the Project
Feel free to add more functionalities, such as:

- Real-time text input for generating images on the fly.
- Integration with web frameworks for an interactive user interface.
- More complex transformations and effects.
