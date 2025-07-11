import os
import base64
import json
from huggingface_hub import InferenceClient

# ------------------- SETTINGS -------------------
PARENT_FOLDER = r"D:\Projects\OMR\new_abhigyan\debugging\annotate_Test_Series\OMR"
CATEGORIES = {
    "reg_no": "Registration Number",
    "roll_no": "Roll Number",
    "booklet_no": "Question Booklet Number"
}
OUTPUT_JSON = r"D:\Projects\OMR\new_abhigyan\debugging\annotate_Test_Series\output.json"

client = InferenceClient(
    provider="hyperbolic",
    api_key="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",
)
# ------------------------------------------------

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"[ERROR] File not found: {image_path}")
        return None

def query_model(encoded_image, category):
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Give me ONLY the handwritten data in the format: {{"{category}":"<handwritten data or None>"}}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Model call failed: {e}")
        return None

def clean_and_parse_json(response_str):
    if response_str.startswith("```json") and response_str.endswith("```"):
        response_str = response_str[len("```json"):-len("```")].strip()

    try:
        return json.loads(response_str)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        print(f"[RAW]: {response_str}")
        return None

# Dictionary to hold final output
final_result = {}

# Process each category folder
for category_folder, label in CATEGORIES.items():
    folder_path = os.path.join(PARENT_FOLDER, category_folder)

    if not os.path.isdir(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}, skipping...")
        continue

    for image_file in os.listdir(folder_path):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, image_file)
        encoded_image = encode_image_to_base64(image_path)

        if not encoded_image:
            continue

        print(f"[INFO] Processing: {image_file} ({category_folder})")
        response = query_model(encoded_image, category_folder)

        if not response:
            continue

        parsed = clean_and_parse_json(response)
        if not parsed or category_folder not in parsed:
            print(f"[WARNING] No valid data found for: {image_file}")
            continue

        value = parsed[category_folder]

        # Initialize entry if not present
        if image_file not in final_result:
            final_result[image_file] = {}

        final_result[image_file][label] = value

# Save the aggregated result
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_result, f, indent=2, ensure_ascii=False)

print(f"\n✅ JSON saved at: {OUTPUT_JSON}")