import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoProcessor

# Function to run the model on an example
def run_example(task_prompt, text_input, image_path, model_dir, device):
    # Load the model and processor from the specified directory
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    # Load and process the image
    image = Image.open(image_path).convert("RGB")

    prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return image, parsed_answer

def main(task_prompt, text_input, image_path, model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image, result = run_example(task_prompt, text_input, image_path, model_dir, device)
    
    # Display the image and result
    plt.figure(figsize=(12, 6))

    # Show image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')

    # Show result on the right
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, result, wrap=True, horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Florence-2 model on an example.")
    parser.add_argument("--task_prompt", type=str, required=True, help="Task prompt for the model.")
    parser.add_argument("--text_input", type=str, required=True, help="Text input for the model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the custom trained model.")

    args = parser.parse_args()
    main(task_prompt=args.task_prompt, text_input=args.text_input, image_path=args.image_path, model_dir=args.model_dir)
