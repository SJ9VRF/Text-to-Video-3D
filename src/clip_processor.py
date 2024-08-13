# Class for CLIP processing
import clip
import torch
from PIL import Image

class CLIPProcessor:
    def __init__(self, device='cuda'):
        """
        Initialize the CLIPProcessor with the specified device.
        :param device: The device to load the CLIP model onto ('cuda' or 'cpu').
        """
        self.device = device
        self.model, self.transform = clip.load('ViT-B/32', device=self.device)

    def preprocess_text(self, text_prompts):
        """
        Preprocess text prompts into tensors that CLIP can process.
        :param text_prompts: List of text strings.
        :return: Tensor of tokenized text.
        """
        return clip.tokenize(text_prompts).to(self.device)

    def encode_prompts(self, text_prompts):
        """
        Encode text prompts into feature vectors using CLIP.
        :param text_prompts: List of text strings to encode.
        :return: Tensor containing the encoded feature vectors.
        """
        text_tokens = self.preprocess_text(text_prompts)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features

    def preprocess_image(self, image_path):
        """
        Preprocess an image for CLIP processing.
        :param image_path: Path to the image file.
        :return: Preprocessed image tensor.
        """
        image = Image.open(image_path)
        return self.transform(image).unsqueeze(0).to(self.device)

    def encode_image(self, image_tensor):
        """
        Encode an image into feature vectors using CLIP.
        :param image_tensor: Tensor of the preprocessed image.
        :return: Tensor containing the encoded image features.
        """
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        return image_features

