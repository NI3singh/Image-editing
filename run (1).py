# import os
# from PIL import Image
# import torch
# from diffusers import QwenImageEditPipeline

# local_model_path = "model/Qwen-Image-Edit" 

# print(f"🚀 Loading pipeline from local path: {local_model_path}")

# # Load the pipeline from your local files
# pipeline = QwenImageEditPipeline.from_pretrained(local_model_path)
# print("✅ Pipeline loaded successfully.")

# pipeline.to("cuda")

# pipeline.set_progress_bar_config(disable=None) 

# # Load your input image
# image = Image.open("./input.png").convert("RGB")

# # Define your editing prompt
# prompt = "Remove the person in the background and change the sky to a beautiful sunrise, with the sun just rising over the horizon."

# # Set up the generation parameters
# inputs = {
#     "image": image,
#     "prompt": prompt,
#     "generator": torch.manual_seed(0), # Use a seed for reproducible results
#     "true_cfg_scale": 4.0,
#     "negative_prompt": " ", # You can add things you don't want to see here
#     "num_inference_steps": 50,
# }

# print("🧠 Generating the edited image...")

# # Run the pipeline
# with torch.inference_mode():
#     output = pipeline(**inputs)
#     output_image = output.images[0]
    
#     # Save the final image
#     output_image.save("output_image_edit.png")
#     print(f"🎉 Success! Image saved at: {os.path.abspath('output_image_edit.png')}")




import os
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline
import gradio as gr

# --- 1. Load the Model (runs only once at startup) ---
local_model_path = "model/Qwen-Image-Edit" 

print(f"🚀 Loading pipeline from local path: {local_model_path}")

try:
    pipeline = QwenImageEditPipeline.from_pretrained(local_model_path)
    pipeline.to("cuda")
    print("✅ Pipeline loaded and moved to GPU successfully.")
except Exception as e:
    print(f"❌ Failed to load the model. Error: {e}")
    exit()

# --- 2. Define the Core Image Editing Function ---
def edit_image(image, prompt): # progress tracker is removed
    """
    Takes a PIL Image and a text prompt, and returns the edited PIL Image.
    NOTE: This function currently only uses the first image, as per the Qwen model's design.
    """
    if image is None:
        raise gr.Error("Please upload an input image.")
    if not prompt:
        raise gr.Error("Please provide a prompt.")

    print("🧠 Generating the edited image...")
    
    # The callback logic has been removed as it is not supported by this pipeline.
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
    
    print("🎉 Generation complete!")
    return output_image

# --- 3. Build the Gradio Web Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Qwen Image Edit Interface")
    gr.Markdown("Upload an image, describe the change you want, and click Generate!")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            
            # Note: The current model logic does not use this second image.
            image_input_2 = gr.Image(type="pil", label="Second Input Image (Optional)")
            
            prompt_input = gr.Textbox(label="Prompt", placeholder="e.g., Change the rabbit's color to purple...")
            generate_button = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            image_output = gr.Image(type="pil", label="Edited Image")

    generate_button.click(
        fn=edit_image,
        # IMPORTANT: We only pass the FIRST image (image_input) to the function.
        inputs=[image_input, prompt_input],
        outputs=[image_output]
    )

# --- 5. Launch the App ---
print("🚀 Launching Gradio app...")
demo.launch(server_name="0.0.0.0", share=True)