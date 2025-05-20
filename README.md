#  Classifier Performance Comparison
---
## Introduction

There are multiple types of classifier models used in image classification; oftentimes, comparisons will be done on different datasets with different metrics used to evaluate the models' performance.

One method used here is to compare the attention mechanisms used in two different models.  Attention allows for neural networks to focus on specific characteristics in an image that is considered significant for its classification.
Self-attention in each segment of an image identifies the inter-relationships between them, calculating values that shift layer to layer.
The final layer is often best representative of the focus of the model when used in classifying an image.
In the two models below, that is represented in form of an RGB heat map.

The two model types compared here are Vision Transformers and Residual Networks.
Vision Transformers breaks the trend of recurrent neural networks and relies on attention to describe global relationships between input and output instead of recurrence.
Residual Networks use residual functions with reference to the layer inputs to create a residual mapping.
The residual value for layers with less relevance to the residual mapping will be near zero and will be skipped. 

### Example from Google Colab

Install the modules needed
```python
!pip install timm
!pip install grad-cam
```
Import the images.

In this example, they are currently set to import from a google drive, but with some alteration, links or arrays of loaded images from a local drive can be substituted in.
```python
from google.colab import drive
drive.mount('/content/drive',force_remount=True)
input_file = '/content/drive/My Drive/Colab Notebooks/Lab3/cutegoat.jpg'
input_file2 = '/content/drive/My Drive/Colab Notebooks/Lab3/catpic.jpg'
input_file3 = '/content/drive/My Drive/Colab Notebooks/Lab3/guppynotreally.jpg'
input_file4 = '/content/drive/My Drive/Colab Notebooks/Lab3/womentour.jpg'
input_file5 = '/content/drive/My Drive/Colab Notebooks/Lab3/titanic.jpg'
input_file6 = '/content/drive/My Drive/Colab Notebooks/Lab3/alien.jpg'
```
#### Visual Transformer
Define the functions used for visualizing the attention maps in the visual transformer.  There are two functions: one for visualizing the attention maps for each layer of the model, and the other for overlaying the final layer's attention map on the image.
```python
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from IPython.display import display

##Functions for Attention Maps

# Visualize total layers attention maps
def visualize_attention(attn_maps_np):
    num_layers = len(attn_maps_np)
    num_heads = attn_maps_np[0].shape[0]

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 2, num_layers * 2))

    for i in range(num_layers):
        for j in range(num_heads):
            ax = axes[i, j] if num_layers > 1 else axes[j]
            attn_map = attn_maps_np[i][j]  # shape: [num_tokens, num_tokens]
            ax.imshow(attn_map, cmap='viridis')
            ax.set_title(f"L{i+1}H{j+1}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# Visualize last layer of attention maps
def show_attention_overlay(image_tensor, attn_map, patch_size=16, alpha=0.6):
    """
    Overlay attention map on top of the input image.
    """
    image = to_pil_image(image_tensor.squeeze(0).cpu()).convert("RGB")
    image_np = np.array(image)

    # Use only the class token's attention to other tokens (shape: [num_tokens])
    # In ViT, first token (index 0) is class token, so take its attention
    cls_attn = attn_map[:, 0, 1:]  # Exclude the class token itself

    # Assume square grid of patches
    num_patches = cls_attn.shape[-1]
    grid_size = int(np.sqrt(num_patches))
    cls_attn_map = cls_attn.reshape(grid_size, grid_size)

    # Normalize attention map
    cls_attn_map -= cls_attn_map.min()
    cls_attn_map /= cls_attn_map.max()

    # Resize attention map to image resolution
    cls_attn_map_resized = cv2.resize(cls_attn_map, (image_np.shape[1], image_np.shape[0]))
    cls_attn_map_color = cv2.applyColorMap(np.uint8(255 * cls_attn_map_resized), cv2.COLORMAP_JET)
    cls_attn_map_color = cv2.cvtColor(cls_attn_map_color, cv2.COLOR_BGR2RGB)

    # Overlay attention on original image
    overlay = (1 - alpha) * image_np + alpha * cls_attn_map_color
    overlay = overlay.astype(np.uint8)

    # Show result
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Class Token Attention Overlay")
    plt.show()
```
Load the vision transformer function
```
#Vision Transformer
import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import urllib
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import requests
from transformers import ViTImageProcessor, ViTForImageClassification
import requests

#Vision Transformer 1
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

#Vision Transformer 2 - secondary model for testing - uncomment if interested in testing
#processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
#model1 = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
```
Create a custom forward function for the transformer blocks.  This will be used to define how the attention scores will be extracted and saved.
```python
for blk in model.blocks:
    # Provide default drop_path if it doesn't exist
    if not hasattr(blk, 'drop_path'):
        blk.drop_path = nn.Identity()

    def new_forward(self, x):
        B, N, C = x.shape

        qkv = self.attn.qkv(x).reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.attn.scale
        attn_weights = attn_scores.softmax(dim=-1)  # (B, num_heads, N, N)
        self.attn_weights = attn_weights  # Store for external use

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        attn_output = self.attn.proj(attn_output)

        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return self.norm1(x)

    # Inject our forward method into each block
    blk.forward = new_forward.__get__(blk, type(blk))
```
Define the image processing function using the ViT model.
Load and pre-process the image by resizing it to 224x224 as the model requires and converting it to a tensor.
```python
def ImageProcess(input_file):
  image = Image.open(input_file).convert("RGB")
  transform = T.Compose([
      T.Resize((224, 224)),
      T.ToTensor(),
      T.Normalize((0.5,), (0.5,))
  ])
  input_tensor = transform(image).unsqueeze(0)

  attention_maps = []
```
Run the model with transformed image.
```python
  # Forward pass
  with torch.no_grad():
      out = model(input_tensor)
      #model1(input_tensor)
```
Extract the stored attention maps, display the original image for comparison and average the last layer attention map to fit the 224x224 resized image.
```python
  # Extract stored attention maps
  attn_maps_np = [blk.attn_weights.squeeze(0).cpu().numpy() for blk in model.blocks]

  #Display original resized image
  display(image.resize(( int(image.width * 0.15), int(image.height * 0.15))))

  #Apply this to the final layer attention map
  last_layer_attn = attn_maps_np[-1]  # [num_heads, num_tokens, num_tokens]
  #This is the portion where the attention is averaged in the last layer before applying to an image
  avg_attn = np.mean(last_layer_attn, axis=0, keepdims=True)  # [1, num_tokens, num_tokens]
```
Use the ImageNet classes database to predict possible classes the images belongs to based on the calculated probabilities from the ViT model.  Print the top three probabilities; this number can be altered as desired by the user.
```python
  # ViT1 model predicts one of the 1000 ImageNet classes
  probabilities = torch.nn.functional.softmax(out[0], dim=0)
  # get imagenet class mappings
  url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
  urllib.request.urlretrieve(url, filename) 
  with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
  top3_prob, top3_catid = torch.topk(probabilities, 3) #can change the number for number of predictions
  for i in range(top3_prob.size(0)):
    print(categories[top3_catid[i]], top3_prob[i].item())
```
Uncomment this section to try a different ViT predictor
```python
  ##Vision Transformer 2 model predictions
  #inputs = processor(images=image, return_tensors="pt")
  #outputs = model1(**inputs)
  #logits = outputs.logits
  #predicted_class_idx = logits.argmax(-1).item()
  #print("Predicted class:", model1.config.id2label[predicted_class_idx])
```
Uncomment this section to see the total attention maps for each layer
```python
  #visualize_attention(attn_maps_np)
```
Display the final layer attention overlay.
```python
  show_attention_overlay(input_tensor, avg_attn)
```
Run the Vision Transformer image processing function
```python
ImageProcess(input_file)
ImageProcess(input_file2)
ImageProcess(input_file3)
ImageProcess(input_file4)
ImageProcess(input_file5)
ImageProcess(input_file6)
```
Here is an example of how the images should appear.
[insert image]
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

#### Residual Network
Load the ResNet modules and instantiate the pre-trained ResNet50 model.
```python
import torch
import torch.nn as nn
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50, decode_predictions

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model2 = resnet50(pretrained=True)
target_layers = [model2.layer4[-1]]
```
Define the second Image Processing function for running the ResNet50 model.
Specify the targets the Grad-CAM module should use for generation the gradient CAM.
Load and pre-process the image similarly to the ViT model.
```python
def ImageProcess2(input_file):
   #Specify target to generate CAM for
  targets = [ClassifierOutputTarget(281)]

  # Load and preprocess image
  image = Image.open(input_file).convert("RGB")
  rgb_img = cv2.imread(input_file, 1)[:, :, ::-1]
  rgb_img = np.float32(rgb_img) / 255
  rgb_img = cv2.resize(rgb_img, (224, 224))

  transform = T.Compose([
      T.Resize((224, 224)),
      T.ToTensor(),
      T.Normalize((0.5,), (0.5,))
  ])
  input_tensor = transform(image).unsqueeze(0)
```
Run the function and use the output probabilities and the same ImageNet classes database to generate a set of predictions.
```python
# Predict the subject of the image
  with torch.no_grad():
      out = model2(input_tensor)
  probabilities = torch.nn.functional.softmax(out[0], dim=0)
  url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
  urllib.request.urlretrieve(url, filename) 
  with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

  # Print top categories per image
  top_prob, top_catid = torch.topk(probabilities, 1)
  for i in range(top_prob.size(0)):
    print(categories[top_catid[i]], top_prob[i].item())
```
Run a forward pass using GradCAM and generate a CAM visualization overlaid on the original image.
Plot the visualization.
```python
  #Model Fwd Pass
  with GradCAM(model2, target_layers) as cam:
    grayscale_cam = cam(input_tensor, targets)
    grayscale_cam = grayscale_cam[0, :]
    #grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    model_outputs = cam.outputs

  # plot last layer of attention map
  plt.figure(figsize=(6, 6))
  plt.imshow(visualization)
  plt.axis("off")
  plt.title("Class Token Attention Overlay")
  plt.show()
```
Run the second image processing function using ResNet50.
```python
ImageProcess2(input_file)
ImageProcess2(input_file2)
ImageProcess2(input_file3)
ImageProcess2(input_file4)
ImageProcess2(input_file5)
ImageProcess2(input_file6)
```
Here is an example of the activation map visualized using GradCAM for several images.


 version/2/1/code_of_conduct/code_of_conduct.md) &bull; [MIT License](https://gh.io/mit)
