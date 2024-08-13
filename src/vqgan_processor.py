# Class for VQGAN processing
from torchvision import transforms
import torch

from torch import nn
import torch
from torchvision import transforms
from omegaconf import OmegaConf
from taming.models import vqgan

class VQGANProcessor:
    def __init__(self, config_path, checkpoint_path):
        self.model = self.load_vqgan_model(config_path, checkpoint_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def load_vqgan_model(self, config_path, checkpoint_path):
        """
        Load and return the VQGAN model from the specified configuration and checkpoint.
        """
        config = OmegaConf.load(config_path)
        model = vqgan.VQModel(**config.model.params)
        model.init_from_ckpt(checkpoint_path)
        return model

    def encode_image(self, image_tensor):
        """
        Encode an image tensor to a latent representation using the VQGAN model's encoder.
        """
        with torch.no_grad():
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            z, _, _ = self.model.encode(image_tensor)
            return z

    def decode_image(self, z):
        """
        Decode a latent representation to an image using the VQGAN model's decoder.
        """
        with torch.no_grad():
            z = z.cuda() if torch.cuda.is_available() else z
            image = self.model.decode(z)
            image = (image / 2 + 0.5).clamp(0, 1)  # Normalize the image to [0, 1]
            return image

    def synthesize_image(self, z):
        """
        Synthesize an image from the latent vector z using the VQGAN model.
        """
        return self.decode_image(z)
