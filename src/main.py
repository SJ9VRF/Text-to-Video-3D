# Main script to run the synthesis process

import torch
from PIL import Image
from src.clip_processor import CLIPProcessor
from src.vqgan_processor import VQGANProcessor
from src.animation import Animation

def main():
    # Specify the device to use for processing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize processors
    clip_processor = CLIPProcessor(device=device)
    vqgan_processor = VQGANProcessor(
        config_path='path/to/vqgan/config.yaml',
        checkpoint_path='path/to/vqgan/checkpoint.ckpt'
    )

    # Initialize animation system
    animation = Animation(vqgan_processor, clip_processor, device)

    # Define text prompts for guiding the image synthesis
    text_prompts = [
        "A futuristic city skyline at sunset, highly detailed, digital art",
        "A forest scene with ethereal lights and fog, digital art"
    ]

    # Example of transformations (if any)
    transformations = {
        'angle': 15,    # Rotate the image by 15 degrees
        'zoom': 1.1,    # Zoom in slightly
        'tx': 10,       # Translate 10 pixels to the right
        'ty': -5        # Translate 5 pixels up
    }

    # Generate frames based on text prompts
    for i, prompt in enumerate(text_prompts):
        print(f"Generating frame for prompt: {prompt}")
        frame = animation.generate_frame([prompt], transformations=transformations)
        frame.save(f'output/frame_{i}.png')
        print(f"Frame {i} saved.")

    print("All frames have been generated and saved.")

if __name__ == "__main__":
    main()

