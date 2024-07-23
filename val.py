import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Function to run the model on an example
def run_example(task_prompt, text_input, image_path, model_name, revision, device):
    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, revision=revision).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, revision=revision)

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
    return parsed_answer

def main(task_prompt, text_input, image_path, model_name, revision):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_example(task_prompt, text_input, image_path, model_name, revision, device)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Florence-2 model on an example.")
    parser.add_argument("--task_prompt", type=str, required=True, help="Task prompt for the model.")
    parser.add_argument("--text_input", type=str, required=True, help="Text input for the model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_name", type=str, default="microsoft/Florence-2-base-ft", help="Model name or path.")
    parser.add_argument("--revision", type=str, default="refs/pr/6", help="Model revision.")

    args = parser.parse_args()
    main(task_prompt=args.task_prompt, text_input=args.text_input, image_path=args.image_path, model_name=args.model_name, revision=args.revision)
