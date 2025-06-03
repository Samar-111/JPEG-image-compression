import torch
import torch.nn.functional as F
from torchvision import transforms
from compressai.zoo import bmshj2018_factorized
from PIL import Image
import matplotlib.pyplot as plt
import os
from google.colab import files

# Function to display images
def display_images(original_img, compressed_img_tensor):
    print("Displaying images...")
    try:
        # Convert tensors to PIL Images for display
        original_img = transforms.ToPILImage()(original_img.squeeze(0).cpu())
        compressed_img = transforms.ToPILImage()(compressed_img_tensor.squeeze(0).cpu())

        # Plot the images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img)
        axes[0].axis("off")
        axes[0].set_title("Original Image")

        axes[1].imshow(compressed_img)
        axes[1].axis("off")
        axes[1].set_title("Compressed Image")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in displaying images: {e}")

# Function to calculate PSNR
def calculate_psnr(original_image, compressed_image):
    print("Calculating PSNR...")
    if original_image.shape != compressed_image.shape:
        compressed_image = F.interpolate(compressed_image, size=original_image.shape[2:], mode="bilinear", align_corners=False)

    mse = torch.mean((original_image - compressed_image) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 1.0  # Assuming normalized [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

# Main Function
def main():
    print("Starting image compression process...")

    # Upload image via Colab file uploader
    uploaded = files.upload()
    if not uploaded:
        print("No image uploaded.")
        return

    image_path = list(uploaded.keys())[0]
    print(f"Uploaded image: {image_path}")

    # Load the image and preprocess
    print("Loading image...")
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    original_img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print(f"Image loaded with shape: {original_img_tensor.shape}")

    # Load the pre-trained CompressAI model
    print("Loading pre-trained CompressAI model...")
    model = bmshj2018_factorized(quality=3, pretrained=True).eval()

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    original_img_tensor = original_img_tensor.to(device)

    # Perform image compression
    print("Compressing image...")
    with torch.no_grad():
        try:
            output = model(original_img_tensor)
            compressed_img_tensor = output["x_hat"].clamp(0, 1)
            print("Compression successful.")
        except Exception as e:
            print(f"Error during compression: {e}")
            return

    # Save compressed image
    print("Saving compressed image...")
    compressed_img = transforms.ToPILImage()(compressed_img_tensor.squeeze(0).cpu())
    save_path = "compressed_output.jpg"
    compressed_img.save(save_path)
    print(f"Compressed image saved at: {save_path}")

    # Display images
    display_images(original_img_tensor, compressed_img_tensor)

    # Calculate and display PSNR
    psnr = calculate_psnr(original_img_tensor, compressed_img_tensor)
    print(f"PSNR: {psnr:.2f} dB")

if _name_ == "_main_":
    main()