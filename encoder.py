import streamlit as st
import torch
import timm
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from sklearn.preprocessing import normalize


device = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureExtractor:
    def __init__(self, modelname):
        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.to(device)
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]
        # easier way to get model config
        config = resolve_model_data_config(self.model)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, image: Image):
        # Preprocess the input image
        input_image = image.convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension and put it in  right device
        input_tensor = input_image.unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector based on runtime
        if device == "cuda":
            feature_vector = output.squeeze().cpu().numpy()
        else:
               feature_vector = output.squeeze().numpy() 

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


# getting model dimension
class ModelDim(FeatureExtractor):
    def __init__(self, modelname):
        super().__init__(modelname)
    def get_dim(self):
        with torch.no_grad():
            return self.model(self.preprocess(torch.randn([1, 3, 10, 10]).to(device))).shape[-1]

# @st.cache_resource
# def load_model(image_encoder):
#     return image_encoder.model
