import torch
from diffusers import QwenImageEditPipeline
from huggingface_hub import snapshot_download
import os

# The official Hugging Face ID for the model
model_id = "Qwen/Qwen-Image-Edit"

local_model_path = "model/Qwen-Image-Edit" 

print(f"🚀 Starting download for model: {model_id}")


try:

    print("Downloading model... (This may take a very long time)")
    snapshot_download(repo_id=model_id, local_dir=local_model_path, resume_download=True)
    print("✅ Model downloaded successfully.")

    print("Loading pipeline from local path...")
    pipeline = QwenImageEditPipeline.from_pretrained(local_model_path)

    pipeline.to(torch_dtype=torch.bfloat16)
    pipeline.to("cuda") 
    
    print("✅ Pipeline loaded and moved to GPU successfully!")
    print("\n🎉 Setup complete! The model is ready to be used.")

except Exception as e:
    print(f"❌ An error occurred: {e}")
    import traceback
    traceback.print_exc()