# Class for animation generation
import torch
from PIL import Image
import numpy as np
from src.transform_utils import TransformUtils

class Animation:
    def __init__(self, vqgan_processor, clip_processor, device):
        """
        Initialize the Animation class with VQGAN and CLIP processors.
        :param vqgan_processor: An instance of VQGANProcessor.
        :param clip_processor: An instance of CLIPProcessor.
        :param device: Torch device to use ('cuda' or 'cpu').
        """
        self.vqgan_processor = vqgan_processor
        self.clip_processor = clip_processor
        self.device = device

    def apply_transformations(self, image, angle=0, zoom=1.0, tx=0, ty=0):
        """
        Apply transformations like rotate, translate, and zoom to the image.
        :param image: Image as a PIL Image.
        :param angle: Angle in degrees for rotation.
        :param zoom: Zoom factor (greater than 1 zooms in, less than 1 zooms out).
        :param tx: Translation in pixels along the x-axis.
        :param ty: Translation in pixels along the y-axis.
        :return: Transformed image as a PIL Image.
        """
        image = np.array(image)
        image = TransformUtils.apply_zoom(image, zoom)
        image = TransformUtils.apply_rotation(image, angle)
        image = TransformUtils.apply_translation(image, tx, ty)
        return Image.fromarray(image)

    def generate_frame(self, text_prompts, init_image=None, transformations=None):
        """
        Generate a single frame based on text prompts and initial image.
        :param text_prompts: List of text prompts to guide the image synthesis.
        :param init_image: Optional initial image to start from (PIL Image).
        :param transformations: Dictionary of transformation parameters.
        :return: Generated frame as a PIL Image.
        """
        if transformations is None:
            transformations = {}

        # Encode prompts using CLIP
        encoded_prompts = self.clip_processor.encode_prompts(text_prompts)
        encoded_prompts = encoded_prompts.to(self.device)

        # If an initial image is provided, use it as the starting point
        if init_image:
            init_image = init_image.convert('RGB')
            init_image = TransformUtils.preprocess(init_image).unsqueeze(0).to(self.device)
            z = self.vqgan_processor.encode_image(init_image)
        else:
            # Start from a random latent space vector
            z = torch.randn([1, 256, 16, 16], device=self.device)  # Assuming the latent dimensions here

        # Apply transformations if any
        if transformations:
            init_image = self.apply_transformations(init_image, **transformations)

        # Synthesize the image using the encoded prompts and latent vector
        synthesized_image = self.vqgan_processor.synthesize_image(z)
        synthesized_image = synthesized_image.cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
        synthesized_image = (synthesized_image * 255).astype(np.uint8)
        return Image.fromarray(synthesized_image)
