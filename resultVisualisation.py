import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# ------------------------- USER SETTINGS -------------------------

# Input Paths
IMAGE_FOLDER = r'D:\Projects\OMR\new_abhigyan\TestSet\TestData\BE_24_Series'
MERGED_CSV_FILE = r'D:\Projects\OMR\new_abhigyan\TestSet\TestData\BE_24_Series\merged_results.csv'
RESULTS_FOLDER = r'D:\Projects\OMR\new_abhigyan\TestSet\Results'

# Font paths
FONT_REGULAR_PATH = r"D:\Projects\OMR\new_abhigyan\debugging\FontFace\SpaceMono-Regular.ttf"
FONT_BOLD_PATH = r"D:\Projects\OMR\new_abhigyan\debugging\FontFace\SpaceMono-Bold.ttf"


# Font sizes
FONT_LABEL_SIZE = 34
FONT_DATA_SIZE = 38
FONT_ANSWER_SIZE = 36

# Layout tuning
COLUMN_GAP = 32
BLOCK_WIDTH = 180  # Width per answer block

# ------------------------------------------------------------------

def draw_label_and_data(draw_obj, x_pos, y_pos, label, data, font_label, font_data):
    draw_obj.text((x_pos, y_pos), label, font=font_label, fill="black")
    y_pos += FONT_LABEL_SIZE + 4
    draw_obj.text((x_pos + 20, y_pos), str(data), font=font_data, fill="black")
    return y_pos + FONT_DATA_SIZE + 12

def format_question_blocks(answers, draw, font_reg, font_bold, x_start, y_start):
    total = len(answers)

    if total <= 10:
        cols = 1
        blocks = [(0, total)]
    elif total <= 30:
        cols = 2
        mid = total // 2
        blocks = [(0, mid), (mid, total)]
    elif total <= 40:
        cols = 2
        blocks = [(0, 10), (10, 20), (20, 30), (30, 40)]
    else:
        cols = 5
        blocks = [(i, i + 10) for i in range(0, total, 10)]

    col_count = 0
    block_x = x_start
    block_y = y_start
    block_spacing_y = 16
    block_spacing_x = BLOCK_WIDTH + COLUMN_GAP

    for i, (start, end) in enumerate(blocks):
        current_y = block_y + (i % 2) * 250 if len(blocks) == 4 else block_y
        if len(blocks) == 4 and i == 2:
            block_x += block_spacing_x
        if len(blocks) != 4 and i != 0 and i % (len(blocks) // cols) == 0:
            block_x += block_spacing_x
            current_y = block_y

        for q_label, opt in answers[start:end]:
            draw.text((block_x, current_y), q_label, font=font_reg, fill="black")
            label_width = draw.textlength(q_label, font=font_reg)
            draw.text((block_x + label_width + 6, current_y), opt, font=font_bold, fill="black")
            current_y += FONT_ANSWER_SIZE + 4

def visualize_results():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    df = pd.read_csv(MERGED_CSV_FILE)

    font_label = ImageFont.truetype(FONT_REGULAR_PATH, FONT_LABEL_SIZE)
    font_data = ImageFont.truetype(FONT_BOLD_PATH, FONT_DATA_SIZE)
    font_answer_regular = ImageFont.truetype(FONT_REGULAR_PATH, FONT_ANSWER_SIZE)
    font_answer_bold = ImageFont.truetype(FONT_BOLD_PATH, FONT_ANSWER_SIZE)

    for _, row in df.iterrows():
        image_name = row['Image Name']
        image_path = os.path.join(IMAGE_FOLDER, image_name)

        if not os.path.isfile(image_path):
            print(f"⚠️ Image not found: {image_name}")
            continue

        question_columns = [col for col in row.index if col.startswith("Q")]
        total_qs = len(question_columns)

        # Determine blocks and dynamic right padding
        if total_qs <= 10:
            blocks = 1
        elif total_qs <= 30:
            blocks = 2
        elif total_qs <= 40:
            blocks = 2  # 2 cols, 4 blocks (2 per col)
        else:
            blocks = 5
        right_padding = (BLOCK_WIDTH * blocks) + 80

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        new_width = width + right_padding

        new_image = Image.new("RGB", (new_width, height), "white")
        new_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(new_image)
        x_start = width + 10
        y_start = int(height * 0.15)

        # Draw all metadata
        y_start = draw_label_and_data(draw, x_start, y_start, "Registration Number:", row.get('RegistrationNumber', ''), font_label, font_data)
        y_start = draw_label_and_data(draw, x_start, y_start, "Roll Number:", row.get('RollNumber', ''), font_label, font_data)
        y_start = draw_label_and_data(draw, x_start, y_start, "Booklet Number:", row.get('BookletNumber', ''), font_label, font_data)

        y_start += 20
        y_start = draw_label_and_data(draw, x_start, y_start, "ICR Registration Number:", row.get('ICR RegistrationNumber', ''), font_label, font_data)
        y_start = draw_label_and_data(draw, x_start, y_start, "ICR Roll Number:", row.get('ICR RollNumber', ''), font_label, font_data)
        y_start = draw_label_and_data(draw, x_start, y_start, "ICR Booklet Number:", row.get('ICR BookletNumber', ''), font_label, font_data)

        # Prepare and draw answers
        answers = [(q + ":", str(row[q])[-1] if pd.notna(row[q]) else '') for q in question_columns]
        answers_start_y = max(y_start + 20, int(height * 0.65))
        format_question_blocks(answers, draw, font_answer_regular, font_answer_bold, x_start, answers_start_y)

        # Save
        save_path = os.path.join(RESULTS_FOLDER, image_name)
        new_image.save(save_path)
        print(f"✅ Saved: {save_path}")

# Run it
if __name__ == "__main__":
    visualize_results()