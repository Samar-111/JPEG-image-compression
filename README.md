📸 JPEG Image Compression using CompressAI
This project demonstrates JPEG-like image compression using a pre-trained deep learning model from the CompressAI library. It allows you to upload an image, compress it using a learned model (bmshj2018-factorized), and evaluate compression quality with PSNR and visual comparison.

🚀 Features
Upload and compress your own image (via Google Colab)

Uses pre-trained neural compression model: bmshj2018-factorized

Displays original and compressed images side by side

Calculates Peak Signal-to-Noise Ratio (PSNR) for quality comparison

Saves compressed image as compressed_output.jpg

🧠 Model Used
Model: bmshj2018-factorized

Source: CompressAI Paper (Balle et al., ICLR 2018)

Trained for: End-to-end neural image compression

🧰 Requirements
Python ≥ 3.6

Torch ≥ 1.7

torchvision

Pillow

matplotlib

CompressAI

Google Colab (for interactive use)

Install required packages locally (if not using Colab):

bash
Copy
Edit
pip install torch torchvision matplotlib pillow compressai

📂 How to Use (Colab)
Clone or upload this script to a Google Colab notebook.

Run the cells — the script will:

Prompt you to upload an image.

Compress the image using the bmshj2018-factorized model.

Display original vs compressed images.

Show PSNR value.

Compressed image will be saved as compressed_output.jpg.

📸 Sample Output
it will give:-
Original Image	Compressed Image

PSNR: 34.89 dB

📜 Code Highlights
Image Upload via google.colab.files.upload()

Model Inference using:

python
Copy
Edit
model = bmshj2018_factorized(quality=3, pretrained=True)
output = model(original_img_tensor)
PSNR Calculation using:

python
Copy
Edit
mse = torch.mean((original - compressed) ** 2)
psnr = 20 * log10(1.0 / sqrt(mse))

⚠️ Notes
This code is optimized for use in Google Colab.

Images are normalized to [0, 1] and output clamped accordingly.

The quality parameter can be adjusted (1 = lowest, 8 = highest).

📄 License
This project is under the MIT License.

Model and code are based on CompressAI, licensed under the Apache License 2.0.

🙌 Acknowledgments
CompressAI by InterDigital Inc.

PyTorch, torchvision, and PIL for core functionalities.

