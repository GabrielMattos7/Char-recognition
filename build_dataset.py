import os
import re

def create_label_files(text_dir, cropped_dir):
    text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])
    all_text = ''
    for text_file in text_files:
        with open(os.path.join(text_dir, text_file), 'r', encoding='utf-8') as f:
            all_text += f.read()

    # all_text = re.sub(r'\s+', '', all_text)  # Remove all whitespace
    all_text = ''.join(char for char in all_text if char.isprintable())  # Remove non-printable characters

    cropped_images = sorted([f for f in os.listdir(cropped_dir) if f.endswith('.png')])

    # Create label files
    text_index = 0
    for image_file in cropped_images:
        if text_index < len(all_text):
            # skipw whitespace
            while all_text[text_index] == " ":
                text_index += 1
                
            label = all_text[text_index]
            
            # Check for double 'f's
            if label == 'f' and text_index + 1 < len(all_text) and all_text[text_index + 1] == 'f':
                label = 'ff'
                text_index += 1  # Skip the next 'f'
            # elif label == 't' and text_index + 1 < len(all_text) and all_text[text_index + 1] == 't':
            #     label = 'tt'
            #     text_index += 1  # Skip the next 'f'
            # elif label == 'r' and text_index + 1 < len(all_text) and all_text[text_index + 1] == 't':
            #     label = 'rt'
            #     text_index += 1  # Skip the next 'f'
            # elif label == 'r' and text_index + 1 < len(all_text) and all_text[text_index + 1] == 'f':
            #     label = 'rf'
            #     text_index += 1  # Skip the next 'f'
            # elif label == 't' and text_index + 1 < len(all_text) and all_text[text_index + 1] == 'w':
            #     label = 'tw'
            #     text_index +=1
            # elif label == 'y' and text_index + 1 < len(all_text) and all_text[text_index + 1] == 'w':
            #     label = 'yw'
            #     text_index +=1
            # elif label == 'f' and text_index + 1 < len(all_text) and all_text[text_index + 1] == 't':
            #     label = 'ft'
            #     text_index +=1
            # elif label == 'r' and text_index + 1 < len(all_text) and all_text[text_index + 1] == 'v':
            #     label = 'rv'
            #     text_index +=1                                                                                          

            label_file = os.path.splitext(image_file)[0] + '.label'
            with open(os.path.join(cropped_dir, label_file), 'w', encoding='utf-8') as f:
                f.write(label)
            text_index += 1
        else:
            print(f"Warning: More images than characters in text. Skipping {image_file}")

if __name__ == "__main__":
    create_label_files("text", "cropped_characters")
