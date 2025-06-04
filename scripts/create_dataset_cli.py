import argparse
import csv
import json
import os
import random
from PIL import Image


def generate_random_question_id(existing_ids):
    while True:
        qid = random.randint(10000, 99999)
        if qid not in existing_ids:
            return qid


def load_existing_ids(dataset_folder):
    ids = set()
    if os.path.isdir(dataset_folder):
        for name in os.listdir(dataset_folder):
            if name.endswith(".json"):
                with open(os.path.join(dataset_folder, name), "r") as f:
                    data = json.load(f)
                ids.add(int(data.get("questionId", 0)))
    return ids


def save_entry(entry, image_path, dataset_folder):
    qid = entry["questionId"]
    os.makedirs(dataset_folder, exist_ok=True)
    json_path = os.path.join(dataset_folder, f"{qid}.json")
    with open(json_path, "w") as f:
        json.dump(entry, f, indent=4)
    Image.open(image_path).convert("RGB").save(
        os.path.join(dataset_folder, f"{qid}.png")
    )


def process_csv(csv_path, dataset_folder):
    existing_ids = load_existing_ids(dataset_folder)
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            qid = generate_random_question_id(existing_ids)
            existing_ids.add(qid)
            entry = {
                "questionId": qid,
                "question": row.get("question", ""),
                "question_types": row.get("question_types", ""),
                "docId": row.get("docId", ""),
                "ucsf_document_id": row.get("ucsf_document_id", ""),
                "ucsf_document_page_no": row.get("ucsf_document_page_no", ""),
                "answers": row.get("answers", ""),
            }
            image_path = row.get("image_path")
            if not image_path:
                continue
            save_entry(entry, image_path, dataset_folder)


def interactive_mode(image_folder, dataset_folder):
    existing_ids = load_existing_ids(dataset_folder)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(("png", "jpg", "jpeg"))]
    image_files.sort()
    prev_question = ""
    prev_types = ""
    for img in image_files:
        path = os.path.join(image_folder, img)
        print(f"\nProcessing {img}")
        qid = generate_random_question_id(existing_ids)
        existing_ids.add(qid)
        question = input(f"Question [{prev_question}]: ") or prev_question
        qtypes = input(f"Question Types [{prev_types}]: ") or prev_types
        doc_id = input("Doc ID: ")
        ucsf_id = input("UCSF Document ID: ")
        page_no = input("UCSF Document Page No: ")
        answers = input("Answers: ")
        entry = {
            "questionId": qid,
            "question": question,
            "question_types": qtypes,
            "docId": doc_id,
            "ucsf_document_id": ucsf_id,
            "ucsf_document_page_no": page_no,
            "answers": answers,
        }
        save_entry(entry, path, dataset_folder)
        prev_question = question
        prev_types = qtypes
        print(f"Saved {qid}\n")


def main():
    parser = argparse.ArgumentParser(description="CLI dataset generator")
    parser.add_argument("--image_folder", help="Folder with images for interactive mode")
    parser.add_argument("--dataset_folder", default="dataset", help="Output dataset folder")
    parser.add_argument("--metadata_csv", help="CSV file with image paths and labels")
    args = parser.parse_args()

    if args.metadata_csv:
        process_csv(args.metadata_csv, args.dataset_folder)
    elif args.image_folder:
        interactive_mode(args.image_folder, args.dataset_folder)
    else:
        parser.error("Specify --metadata_csv or --image_folder")


if __name__ == "__main__":
    main()
