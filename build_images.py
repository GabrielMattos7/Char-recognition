import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_text_image(text_file, output_file, font_size=40, text_color=(0, 0, 0), bg_color=(255, 255, 255), image_size=(800, 600)):
    with open(text_file, 'r') as file:
        text = file.read()

    image = Image.new('RGB', image_size, bg_color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./arial.ttf", font_size)

    max_width = image_size[0] - 20  # 10px padding on each side
    wrapped_text = textwrap.fill(text, width=max_width // (font_size // 2))
    draw.text((10, 10), wrapped_text, font=font, fill=text_color)

    image.save(output_file)

def process_all_text_files(input_dir, output_dir, font_size=20, text_color=(0, 0, 0), bg_color=(255, 255, 255), image_size=(1000, 800)):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):

        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            create_text_image(input_path, output_path, font_size, text_color, bg_color, image_size)

if __name__ == "__main__":
    process_all_text_files("./text", "./output_images")
