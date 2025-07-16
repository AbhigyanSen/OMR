import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# ------------------------- USER SETTINGS -------------------------

# Input Paths
IMAGE_FOLDER = r'D:\Projects\OMR\new_abhigyan\TestSet\TestData\BE_24_Series'
CSV_FILE = r'D:\Projects\OMR\new_abhigyan\TestSet\TestData\BE_24_Series\summary.csv'
RESULTS_FOLDER = r'D:\Projects\OMR\new_abhigyan\TestSet\Results'

# Font paths (provide your actual TTF paths here)
FONT_REGULAR_PATH = r"D:\Projects\OMR\new_abhigyan\debugging\FontFace\SpaceMono-Regular.ttf"
FONT_BOLD_PATH = r"D:\Projects\OMR\new_abhigyan\debugging\FontFace\SpaceMono-Bold.ttf"        

# Font sizes
FONT_LABEL_SIZE = 34
FONT_DATA_SIZE = 38
FONT_ANSWER_SIZE = 36

# Padding
PADDING_RIGHT = 360

# ------------------------------------------------------------------

# Create output folder if not exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_FILE)

# Load fonts
try:
    font_label = ImageFont.truetype(FONT_REGULAR_PATH, FONT_LABEL_SIZE)
    font_data = ImageFont.truetype(FONT_BOLD_PATH, FONT_DATA_SIZE)
    font_answer_regular = ImageFont.truetype(FONT_REGULAR_PATH, FONT_ANSWER_SIZE)
    font_answer_bold = ImageFont.truetype(FONT_BOLD_PATH, FONT_ANSWER_SIZE)
except Exception as e:
    print("Error loading fonts. Check your font paths.")
    raise e

# Process each row
for index, row in df.iterrows():
    image_name = row['Image Name']
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    if not os.path.isfile(image_path):
        print(f"Image not found: {image_name}, skipping.")
        continue

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Add right padding
    new_width = width + PADDING_RIGHT
    new_image = Image.new("RGB", (new_width, height), "white")
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)
    x_start = width + 10

    # --- Registration / Roll / Booklet Numbers ---
    current_y = int(height * 0.30)  # Start at 30% height

    label_spacing = 40
    section_spacing = 60

    # Draw label + bold data for each
    def draw_label_and_data(draw_obj, x_pos, y_pos, label, data):
        draw_obj.text((x_pos, y_pos), label, font=font_label, fill="black")
        y_pos += label_spacing
        draw_obj.text((x_pos, y_pos), str(data), font=font_data, fill="black")
        y_pos += section_spacing
        return y_pos

    current_y = draw_label_and_data(draw, x_start, current_y, "Registration No:", row['RegistrationNumber'])
    current_y = draw_label_and_data(draw, x_start, current_y, "Roll No:", row['RollNumber'])
    current_y = draw_label_and_data(draw, x_start, current_y, "Booklet No:", row['BookletNumber'])


    # --- Questions Section ---
    y_answers_start = max(int(height * 0.60), current_y + 30)
    line_spacing = 32
    num_columns = 2
    column_width = PADDING_RIGHT // num_columns - 10

    # Collect and format answers
    answers = []
    for i in range(1, 41):
        q_col = f"Q{i}"
        val = str(row[q_col])
        option = val[-1] if val else ""
        answers.append((f"Q{i}:", option))

    items_per_col = len(answers) // num_columns
    if len(answers) % num_columns:
        items_per_col += 1

    # Draw in 2 columns
    current_x = x_start
    current_y = y_answers_start
    for i, (q_label, opt) in enumerate(answers):
        # Draw regular part: "Q1:"
        draw.text((current_x, current_y), q_label, font=font_answer_regular, fill="black")

        # Measure width of "Q1:"
        label_width = draw.textlength(q_label, font=font_answer_regular)

        # Draw bold part: "A"
        draw.text((current_x + label_width + 6, current_y), opt, font=font_answer_bold, fill="black")

        current_y += line_spacing

        # Move to second column when full
        if (i + 1) % items_per_col == 0 and (i + 1) < len(answers):
            current_x += column_width
            current_y = y_answers_start

    # Save image
    save_path = os.path.join(RESULTS_FOLDER, image_name)
    new_image.save(save_path)
    print(f"Saved: {save_path}")