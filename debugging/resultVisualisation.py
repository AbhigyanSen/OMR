import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Paths
IMAGE_FOLDER = r'D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series'
CSV_FILE = r'D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\summary.csv'
RESULTS_FOLDER = r'D:\Projects\OMR\new_abhigyan\debugging\Results'

# Create Results folder if not exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_FILE)

# Loop through each row in the CSV
for index, row in df.iterrows():
    image_name = row['Image Name']
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    # Check if image exists
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_name}, skipping.")
        continue

    # Load image
    image = Image.open(image_path)
    image = image.convert("RGB")  # Ensure RGB

    # Add white space to the right
    width, height = image.size
    padding = 400  # You can adjust this width
    new_width = width + padding
    new_image = Image.new("RGB", (new_width, height), "white")
    new_image.paste(image, (0, 0))

    # Prepare text to draw
    draw = ImageDraw.Draw(new_image)
    try:
        # Increased font size for main details and answers
        font_main_label = ImageFont.truetype("arial.ttf", 26) # Slightly smaller for labels
        font_main_data = ImageFont.truetype("arial.ttf", 30) # For the actual numbers
        font_answers = ImageFont.truetype("arial.ttf", 26)
    except:
        # Fallback to default font if arial.ttf is not found
        font_main_label = ImageFont.load_default()
        font_main_data = ImageFont.load_default()
        font_answers = ImageFont.load_default()

    # --- Position Registration/Roll/Booklet Numbers ---
    x_start_right_panel = width + 10 # Starting X position for all text in the right panel
    line_spacing_data = 35 # Spacing between a label and its data
    section_spacing = 45 # Spacing between Registration, Roll, and Booklet sections

    # Initial Y position for the top of the right panel
    current_y = 50 # Start higher up, as requested in previous iterations, this can be fine-tuned.

    # Registration Number
    draw.text((x_start_right_panel, current_y), "Registration Number:", fill="black", font=font_main_label)
    current_y += line_spacing_data
    draw.text((x_start_right_panel, current_y), str(row['RegistrationNumber']), fill="black", font=font_main_data)
    current_y += section_spacing # Move down for next section

    # Roll Number
    draw.text((x_start_right_panel, current_y), "Roll Number:", fill="black", font=font_main_label)
    current_y += line_spacing_data
    draw.text((x_start_right_panel, current_y), str(row['RollNumber']), fill="black", font=font_main_data)
    current_y += section_spacing # Move down for next section

    # Booklet Number
    draw.text((x_start_right_panel, current_y), "Booklet Number:", fill="black", font=font_main_label)
    current_y += line_spacing_data
    draw.text((x_start_right_panel, current_y), str(row['BookletNumber']), fill="black", font=font_main_data)
    current_y += section_spacing # Move down for next section


    # --- Position Answers in Columns (starting after 50% height) ---
    # Calculate the Y position for answers to start after 50% of image height
    y_answers_start = int(height * 0.50) # This makes it dynamic based on image height

    # Ensure answers start below the Registration/Roll/Booklet block if it goes past 50%
    # This scenario is unlikely with the current layout but good for robustness
    if current_y > y_answers_start:
        y_answers_start = current_y + 30 # Add a small buffer if previous block extends beyond 50% mark


    line_spacing_answers = 30 # Increased line spacing for answer lines
    num_columns = 2 # Keeping 2 columns for now
    column_width = padding // num_columns - 10 # Some margin between columns

    # Collect all answer lines first
    answer_lines = []
    for i in range(1, 41): # Assuming Q1 to Q40
        q_col = f"Q{i}"
        answer = row[q_col]
        answer_lines.append(f"{q_col}: {answer}")

    # Determine how many items per column for even distribution
    items_per_column = len(answer_lines) // num_columns
    if len(answer_lines) % num_columns != 0:
        items_per_column += 1

    # Draw answers in columns
    current_x = x_start_right_panel # Re-use x_start for consistency
    current_y = y_answers_start

    for i, line in enumerate(answer_lines):
        draw.text((current_x, current_y), line, fill="black", font=font_answers)
        current_y += line_spacing_answers

        # Move to the next column if current column is full
        if (i + 1) % items_per_column == 0 and (i + 1) < len(answer_lines):
            current_x += column_width
            current_y = y_answers_start # Reset y for the new column


    # Save result
    save_path = os.path.join(RESULTS_FOLDER, image_name)
    new_image.save(save_path)
    print(f"Saved: {save_path}")