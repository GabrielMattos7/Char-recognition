import os
import re

def create_label_files(text_dir, cropped_dir):
    text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])
    all_text = ''
    for text_file in text_files:
        with open(os.path.join(text_dir, text_file), 'r', encoding='utf-8') as f:
            all_text += f.read()

    all_text = re.sub(r'\s+', ' ', all_text)
    all_text = ''.join(char for char in all_text if char.isprintable() or char.isspace())

    cropped_images = sorted([f for f in os.listdir(cropped_dir) if f.endswith('.png')])

    # Create label files
    for i, image_file in enumerate(cropped_images):
        if i < len(all_text):
            label = all_text[i]
            label_file = os.path.splitext(image_file)[0] + '.label'
            with open(os.path.join(cropped_dir, label_file), 'w', encoding='utf-8') as f:
                f.write(label)
        else:
            print(f"Warning: More images than characters in text. Skipping {image_file}")

if __name__ == "__main__":
    text_directory = "./text"
    cropped_directory = "./cropped_characters"
    create_label_files(text_directory, cropped_directory)
