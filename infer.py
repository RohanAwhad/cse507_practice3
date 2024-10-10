import argparse
import torch
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Mask2FormerForUniversalSegmentation  # Ensure this import is correct

# Preprocessing for inference
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def inference_single_image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        logits = outputs.masks_queries_logits
        logits = torch.nn.functional.interpolate(logits, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
        predicted_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return predicted_mask

def visualize_segmentation(image_path: str, mask):
  image = Image.open(image_path).convert("RGB")
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.title("Original Image")
  plt.imshow(image)
  plt.subplot(1, 2, 2)
  plt.title("Predicted Mask")
  plt.imshow(mask, cmap='jet')
  # Save the figure instead of showing it
  save_path = os.path.splitext(image_path)[0] + '_inference.png'
  plt.savefig(save_path)
  plt.close()


def main():
  parser = argparse.ArgumentParser(description="Image Segmentation Inference")
  parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
  parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
  args = parser.parse_args()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_path).to(device).eval()
  predicted_mask = inference_single_image(model, args.image_path, device)
  visualize_segmentation(args.image_path, predicted_mask)

if __name__ == "__main__":
  main()
