# Class for CLIP processing
import clip

class CLIPProcessor:
    def __init__(self, model_path):
        self.model, self.preprocess = clip.load(model_path)

    def encode_prompts(self, prompts):
        # Use CLIP to encode text prompts into feature vectors
        return self.model.encode_text(clip.tokenize(prompts))
