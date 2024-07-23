import os
import json
import random
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PIL import Image

# Function to generate a random question_id
def generate_random_question_id(existing_ids):
    while True:
        question_id = random.randint(10000, 99999)
        if question_id not in existing_ids:
            return question_id

class ImageLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Labeling Tool")
        self.setGeometry(100, 100, 1240, 867)

        self.image_files = []
        self.current_image_index = 0
        self.existing_question_ids = set()
        self.current_image = None
        self.ucsf_document_id = "default_id"
        self.ucsf_document_page_no = "1"
        self.doc_id = "default_doc_id"

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        content_layout = QHBoxLayout()

        # Left side for image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(600, 800)
        self.image_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.image_label)

        # Right side for input fields
        right_layout = QVBoxLayout()

        self.entry_question_id = QLineEdit()
        self.entry_question_id.setReadOnly(True)
        self.entry_question = QLineEdit()
        self.entry_question_types = QLineEdit()
        self.entry_doc_id = QLineEdit(self.doc_id)
        self.entry_ucsf_document_id = QLineEdit(self.ucsf_document_id)
        self.entry_ucsf_document_page_no = QLineEdit(self.ucsf_document_page_no)
        self.entry_answers = QLineEdit()

        right_layout.addWidget(QLabel("Question ID:"))
        right_layout.addWidget(self.entry_question_id)
        right_layout.addWidget(QLabel("Question:"))
        right_layout.addWidget(self.entry_question)
        right_layout.addWidget(QLabel("Question Types:"))
        right_layout.addWidget(self.entry_question_types)
        right_layout.addWidget(QLabel("Doc ID:"))
        right_layout.addWidget(self.entry_doc_id)
        right_layout.addWidget(QLabel("UCSF Document ID:"))
        right_layout.addWidget(self.entry_ucsf_document_id)
        right_layout.addWidget(QLabel("UCSF Document Page No:"))
        right_layout.addWidget(self.entry_ucsf_document_page_no)
        right_layout.addWidget(QLabel("Answers:"))
        right_layout.addWidget(self.entry_answers)

        self.save_button = QPushButton("Save and Next")
        self.save_button.clicked.connect(self.save_data)
        right_layout.addWidget(self.save_button)

        content_layout.addLayout(right_layout)

        main_layout.addLayout(content_layout)

        # Open folder button at the bottom
        open_folder_button = QPushButton("Open Image Folder")
        open_folder_button.clicked.connect(self.open_image_folder)
        main_layout.addWidget(open_folder_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def open_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder_path:
            return

        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if not self.image_files:
            QMessageBox.warning(self, "No Images", "No images found in the selected folder.")
            return

        self.current_image_index = 0

        if not os.path.exists("dataset"):
            os.makedirs("dataset")

        self.existing_question_ids = set()
        for file_name in os.listdir("dataset"):
            if file_name.endswith(".json"):
                with open(os.path.join("dataset", file_name), 'r') as json_file:
                    data = json.load(json_file)
                    self.existing_question_ids.add(data["questionId"])

        self.load_image()

    def load_image(self):
        if self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            self.current_image = Image.open(image_path).convert('RGB')

            qimage = QImage(image_path)
            pixmap = QPixmap.fromImage(qimage).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

            question_id = generate_random_question_id(self.existing_question_ids)
            self.entry_question_id.setText(str(question_id))

            if self.current_image_index > 0:
                self.entry_question.setText(self.previous_question)
                self.entry_question_types.setText(self.previous_question_types)

    def save_data(self):
        question_id = int(self.entry_question_id.text())
        question = self.entry_question.text()
        question_types = self.entry_question_types.text()
        doc_id = self.entry_doc_id.text()
        ucsf_document_id = self.entry_ucsf_document_id.text()
        ucsf_document_page_no = self.entry_ucsf_document_page_no.text()
        answers = self.entry_answers.text()

        if not question or not doc_id or not ucsf_document_id or not ucsf_document_page_no:
            QMessageBox.warning(self, "Input Error", "Please fill in all required fields.")
            return

        self.existing_question_ids.add(question_id)

        json_data = {
            "questionId": question_id,
            "question": question,
            "question_types": question_types,
            "docId": doc_id,
            "ucsf_document_id": ucsf_document_id,
            "ucsf_document_page_no": ucsf_document_page_no,
            "answers": answers
        }

        json_file_path = os.path.join("dataset", f"{question_id}.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        image_file_path = os.path.join("dataset", f"{question_id}.png")
        self.current_image.save(image_file_path)

        self.previous_question = question
        self.previous_question_types = question_types

        self.current_image_index += 1
        if self.current_image_index < len(self.image_files):
            self.load_image()
        else:
            QMessageBox.information(self, "Completed", "All images have been processed.")
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageLabelingApp()
    window.show()
    sys.exit(app.exec())
