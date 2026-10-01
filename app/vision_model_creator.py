import os
import tarfile

from huggingface_hub import snapshot_download

MODEL_DIR = "./model_files"
CODE_DIR = os.path.join(MODEL_DIR, "code")

# 1. Download TrOCR model weights and configuration from Hugging Face
print("Downloading model weights...")
snapshot_download(
    repo_id="microsoft/trocr-base-stage1",
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
)

# 2. Create the code/ directory required by SageMaker
os.makedirs(CODE_DIR, exist_ok=True)

# 3. Write the dedicated inference script inference.py
inference_code = """import io
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def model_fn(model_dir, content=None):
    \"\"\"Load the model when the container starts (runs once).\"\"\"
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.eval()
    return {"model": model, "processor": processor}

def transform_fn(model_dict, input_data, content_type, accept_header):
    \"\"\"Handle an inference request: receive an image and return recognized text.\"\"\"
    model = model_dict["model"]
    processor = model_dict["processor"]

    # Validate the input content type (e.g. image/png, image/jpeg, application/x-image)
    if "image" not in content_type and content_type != "application/x-image":
        raise ValueError(f"Unsupported Content-Type: {content_type}. Please send an image file.")

    # Decode the image from the byte stream
    image = Image.open(io.BytesIO(input_data)).convert("RGB")

    # Prepare inputs and run prediction
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text, "text/plain"
"""

with open(os.path.join(CODE_DIR, "inference.py"), "w", encoding="utf-8") as f:
    f.write(inference_code)

print("Created code/inference.py")

# 4. Package everything into vision-model.tar.gz
print("Packaging vision-model.tar.gz...")
with tarfile.open("vision-model.tar.gz", "w:gz") as tar:
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            # Relative path inside the tar archive
            rel_path = os.path.relpath(full_path, MODEL_DIR)
            tar.add(full_path, arcname=rel_path)

print("Successfully created vision-model.tar.gz ready for upload to S3!")
