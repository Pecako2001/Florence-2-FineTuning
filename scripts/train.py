import os
import torch
import torch.multiprocessing as mp
import argparse
import csv
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoModelForObjectDetection, AutoProcessor
from torch.utils.data import DataLoader
from florence.dataset_utils import (
    load_local_dataset,
    DocVQADataset,
    ObjectDetectionDataset,
)
from tqdm import tqdm
from transformers import AdamW, get_scheduler

# Set multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)


# Collate function for DataLoader
def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers

# Collate function for object detection
def collate_fn_detection(batch, processor, device):
    images, targets = zip(*batch)
    inputs = processor(images=list(images), annotations=list(targets), return_tensors="pt").to(device)
    return inputs

# Function to save losses to CSV and plot them
def save_losses_and_plot(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    
    # Save to CSV
    with open('losses.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        writer.writerows(zip(epochs, train_losses, val_losses))
    
    # Plot and save the graph
    plt.figure()
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_graph.png')
    plt.close()

# Training function
def train_model(train_loader, val_loader, model, processor, device, task_type="DocVQA", epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            if task_type == "DocVQA":
                inputs, answers = batch
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                answers = [str(answer) for answer in answers]
                labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            else:
                inputs = batch
                outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                if task_type == "DocVQA":
                    inputs, answers = batch
                    input_ids = inputs["input_ids"]
                    pixel_values = inputs["pixel_values"]
                    answers = [str(answer) for answer in answers]
                    labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                else:
                    inputs = batch
                    outputs = model(**inputs)
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Average Validation Loss: {avg_val_loss}")

        if epoch % 10 == 0:
            # Save model checkpoint
            output_dir = f"./model_checkpoints/epoch_{epoch + 1}"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

        # Save losses and update the plot
        save_losses_and_plot(train_losses, val_losses)

def main(dataset_folder='dataset', split_ratio=0.8, batch_size=2, num_workers=0, epochs=2, task_type="DocVQA"):
    # Load dataset from local files
    data = load_local_dataset(dataset_folder, task_type=task_type)

    # Check if all entries have images and count them
    train_images_count = sum(1 for entry in data if 'image' in entry)
    print(f"Total images in the dataset: {train_images_count}")

    # Split dataset into training and validation sets
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    print(f"Total images in the training set: {sum(1 for entry in train_data if 'image' in entry)}")
    print(f"Total images in the validation set: {sum(1 for entry in val_data if 'image' in entry)}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor
    if task_type == "ObjectDetection":
        model = AutoModelForObjectDetection.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6').to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6').to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision='refs/pr/6')

    # Create datasets and dataloaders
    if task_type == "ObjectDetection":
        train_dataset = ObjectDetectionDataset(train_data)
        val_dataset = ObjectDetectionDataset(val_data)
        collate = collate_fn_detection
    else:
        train_dataset = DocVQADataset(train_data)
        val_dataset = DocVQADataset(val_data)
        collate = collate_fn

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, processor, device), num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: collate(x, processor, device), num_workers=num_workers)

    # Train the model
    train_model(train_loader, val_loader, model, processor, device, task_type=task_type, epochs=epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Florence-2 model.")
    parser.add_argument("--dataset_folder", type=str, default="dataset", help="Folder containing the dataset.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/validation split ratio.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--task_type", type=str, choices=["DocVQA", "ObjectDetection"], default="DocVQA", help="Task type for fine-tuning.")

    args = parser.parse_args()
    main(dataset_folder=args.dataset_folder, split_ratio=args.split_ratio, batch_size=args.batch_size, num_workers=args.num_workers, epochs=args.epochs, task_type=args.task_type)
