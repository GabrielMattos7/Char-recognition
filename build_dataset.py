import os
import re

def create_label_files(text_dir, cropped_dir):
    text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])
    all_text = ''
    for text_file in text_files:
        with open(os.path.join(text_dir, text_file), 'r', encoding='utf-8') as f:
            all_text += f.read()

    all_text = ''.join(char for char in all_text if char.isprintable())  # Remove non-printable characters

    cropped_images = sorted([f for f in os.listdir(cropped_dir) if f.endswith('.png')])

    text_index = 0
    for image_file in cropped_images:
        if text_index < len(all_text):
            # skip whitespace
            while all_text[text_index] == " ":
                text_index += 1
                
            label = all_text[text_index]
            
            label_file = os.path.splitext(image_file)[0] + '.label'
            with open(os.path.join(cropped_dir, label_file), 'w', encoding='utf-8') as f:
                f.write(label)
            text_index += 1
        else:
            print(f"Warning: More images than characters in text. Skipping {image_file}")

if __name__ == "__main__":
    create_label_files("text", "cropped_characters")
