import argparse
import os
from pdf2image import convert_from_path


def convert_pdf_to_images(pdf_path, output_folder, dpi=300, fmt="png"):
    """Convert all pages of a PDF to images.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Directory where images will be saved.
        dpi (int, optional): Resolution in DPI. Defaults to 300.
        fmt (str, optional): Image format (png or jpeg). Defaults to "png".

    Returns:
        list[str]: Paths to the generated image files.
    """
    os.makedirs(output_folder, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, page in enumerate(pages, start=1):
        image_path = os.path.join(output_folder, f"{base}_{i}.{fmt}")
        page.save(image_path, fmt.upper())
        image_paths.append(image_path)
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Convert PDF pages into images.")
    parser.add_argument("pdf_path", help="Path to the PDF document")
    parser.add_argument("--output_folder", default="images_out", help="Folder to save images")
    parser.add_argument("--dpi", type=int, default=300, help="Output image resolution (DPI)")
    parser.add_argument("--format", choices=["png", "jpeg", "jpg"], default="png", help="Image format")
    args = parser.parse_args()

    images = convert_pdf_to_images(args.pdf_path, args.output_folder, dpi=args.dpi, fmt=args.format)
    print(f"Saved {len(images)} images to {args.output_folder}")
    for img in images:
        print(img)


if __name__ == "__main__":
    main()
