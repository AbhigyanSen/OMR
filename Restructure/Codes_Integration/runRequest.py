import os
import json
import csv
import requests
import re
import copy

# ---------------- Configuration ---------------- #
API_URL = "http://10.4.1.66:8003/predict"

base_folder = r"D:\Projects\OMR\new_abhigyan\Restructure"

omr_template_name = "HSOMR"
date = "23072025"
batch_name = "Batch002"  

ICR_IMAGE_DIR = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, "annotate_" + batch_name, "ICR")

# Output ICR results JSON
ICR_OUTPUT_JSON = os.path.join(ICR_IMAGE_DIR, "ICR_Images.json")

# ED results paths
ED_JSON_PATH = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, "options_" + batch_name, "ed_results.json")
ED_CSV_PATH = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, "options_" + batch_name, "ed_results.csv")

# ---------------- Helper Functions ---------------- #

def collect_icr_images(base_dir):
    """Collect image file paths categorized by subfolder names (used as category)."""
    all_extracted_data = {}
    images_to_process = []

    all_categories = [d for d in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, d))]

    print(f"Detected categories: {all_categories}")

    for category in all_categories:
        category_path = os.path.join(base_dir, category)

        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                full_path = os.path.join(category_path, filename)
                image_name = os.path.splitext(filename)[0]
                images_to_process.append((full_path, image_name, category))

                if image_name not in all_extracted_data:
                    all_extracted_data[image_name] = {cat: "" for cat in all_categories}
            else:
                print(f"Skipping non-image: {filename}")

    return images_to_process, all_extracted_data


def call_icr_api(images_to_process, all_extracted_data):
    """Send each image to the ICR API and store extracted fields."""
    for image_path, image_name, category in images_to_process:
        print(f"\nProcessing: {image_name} | Category: {category}")
        try:
            with open(image_path, 'rb') as f:
                content_type = 'image/png' if image_path.lower().endswith('.png') else 'image/jpeg'
                files = {'image': (os.path.basename(image_path), f, content_type)}
                data = {'category': category}

                response = requests.post(API_URL, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    value = result.get(category, "Error: Missing field in API response")
                    all_extracted_data[image_name][category] = value
                    print(f"  → Extracted: {value}")
                else:
                    error = f"Error: {response.status_code}"
                    all_extracted_data[image_name][category] = error
                    print(f"  → API failure: {error}")

        except Exception as e:
            all_extracted_data[image_name][category] = f"Error: {str(e)}"
            print(f"  → Exception: {e}")
    return all_extracted_data


def save_json(data, path):
    """Save dictionary as formatted JSON."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved ICR data to: {path}")
    except Exception as e:
        print(f"Failed to save JSON: {e}")


def merge_icr_with_ed_results(icr_results, ed_json_path, ed_csv_path,
                              output_json_path=None, output_csv_path=None):
    """Merge ICR results into existing ED JSON and CSV files."""
    if output_json_path is None:
        output_json_path = ed_json_path.replace(".json", "_with_icr.json")
    if output_csv_path is None:
        output_csv_path = ed_csv_path.replace(".csv", "_with_icr.csv")

    # JSON Merge
    print(f"\nMerging ICR into JSON: {ed_json_path}")
    try:
        with open(ed_json_path, 'r', encoding='utf-8') as f:
            ed_data = json.load(f)

        for image_name, icr_fields in icr_results.items():
            key = f"{image_name}.jpg"  # Assumes .jpg; customize if needed
            if key in ed_data:
                for field, value in icr_fields.items():
                    ed_data[key][f"ICR_{field}"] = value
            else:
                ed_data[key] = {f"ICR_{k}": v for k, v in icr_fields.items()}
                print(f"  → Added new entry: {key}")

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(ed_data, f, indent=4, ensure_ascii=False)
        print(f"Saved merged JSON to: {output_json_path}")
    except Exception as e:
        print(f"Failed JSON merge: {e}")

    # CSV Merge
    print(f"\nMerging ICR into CSV: {ed_csv_path}")
    try:
        with open(ed_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            fieldnames = reader[0].keys() if reader else []

        # Get all ICR fields
        icr_fieldnames = {f"ICR_{field}" for v in icr_results.values() for field in v.keys()}
        new_fields = sorted(icr_fieldnames - set(fieldnames))
        final_fields = list(fieldnames) + new_fields

        updated_rows = []
        for row in reader:
            image_name = row.get("Image Name")
            img_key = os.path.splitext(image_name)[0] if image_name else ""
            icr_data = icr_results.get(img_key, {})

            for field, value in icr_data.items():
                row[f"ICR_{field}"] = value
            updated_rows.append(row)

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_fields)
            writer.writeheader()
            writer.writerows(updated_rows)

        print(f"Saved merged CSV to: {output_csv_path}")
    except Exception as e:
        print(f"Failed CSV merge: {e}")
        
# def clean_icr_json(input_json_path):
#     """
#     Clean special characters from all field values in the ICR JSON.
#     Updates the same JSON file in-place.
#     """
#     try:
#         with open(input_json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         for key, fields in data.items():
#             for field, value in fields.items():
#                 # Keep only alphanumeric characters
#                 cleaned_value = re.sub(r'[^0-9A-Za-z]', '', str(value))
#                 data[key][field] = cleaned_value

#         with open(input_json_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=4, ensure_ascii=False)

#         print(f"ICR JSON cleaned and updated in-place: {input_json_path}")

#     except Exception as e:
#         print(f"Failed to clean JSON: {e}")

# Generate a generalized JSON file for the batch ---------------------------------------------------------
def merge_icr_fields_with_generalized_json(base_json_path, icr_json_path, output_path,
                                           key_fields_json=None, human_readable=True):
    # Load JSONs
    with open(base_json_path, "r", encoding="utf-8") as f:
        gen_data = json.load(f)
    with open(icr_json_path, "r", encoding="utf-8") as f:
        icr_data = json.load(f)
    if not key_fields_json:
        key_fields_json = os.path.join(base_folder, "Annotations", omr_template_name, "key_fields.json")
    with open(key_fields_json, "r", encoding="utf-8") as f:
        key_fields = json.load(f)

    for image in gen_data.get("IMAGES", []):
        img_name = os.path.splitext(os.path.basename(image["IMAGENAME"]))[0]
        icr_fields = icr_data.get(img_name, {})

        new_field_list = []
        for field in image["FIELDS"]:
            # keep original field
            new_field_list.append(copy.deepcopy(field))

            # check if field name is one of our keys (like key0, key1...)
            if field["FIELD"] in key_fields:
                icr_field_name = key_fields[field["FIELD"]]
            #     icr_value = icr_fields.get(icr_field_name, "")

            #     confidence = "100" if icr_value and str(icr_value).strip() and "error" not in str(icr_value).lower() else ""
            #     success = "Y" if confidence == "100" else "N"

            #     icr_field = copy.deepcopy(field)
            #     icr_field["FIELD"] = f"{field['FIELD']} ICR"
            #     icr_field["FIELDDATA"] = icr_value
            #     icr_field["CONFIDENCE"] = confidence
            #     icr_field["SUCCESS"] = success
            #     new_field_list.append(icr_field)
            
            icr_value = icr_fields.get(icr_field_name, "")

            # --- Error Handling ---
            if icr_value and "error" in str(icr_value).lower():
                icr_value = ""  # Blank if error

            confidence = "100" if icr_value.strip() else ""
            success = "Y" if confidence == "100" else "N"

            icr_field = copy.deepcopy(field)
            icr_field["FIELD"] = f"{field['FIELD']} ICR"
            icr_field["FIELDDATA"] = icr_value
            icr_field["CONFIDENCE"] = confidence
            icr_field["SUCCESS"] = success
            new_field_list.append(icr_field)


        # update field sequence
        for idx, f in enumerate(new_field_list, start=1):
            f["SEQUENCE"] = idx

        # ------- Human Readable Conversion (New Block) -------
        if human_readable:
            for f in new_field_list:
                original_name = f["FIELD"]
                # For key fields
                if original_name in key_fields:
                    f["FIELD"] = key_fields[original_name]
                elif original_name.endswith(" ICR") and original_name[:-4] in key_fields:
                    f["FIELD"] = key_fields[original_name[:-4]] + " ICR"
                # For question fields
                elif original_name.startswith("question_"):
                    q_num = original_name.split("_")[1]
                    f["FIELD"] = f"Q{q_num}"
        # ------------------------------------------------------

        image["FIELDS"] = new_field_list

    # save output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gen_data, f, indent=4)

    print(f"✅ Dynamic ICR keys merged{' with human-readable names' if human_readable else ''} and saved to {output_path}")


# ---------------- Main Orchestration ---------------- #

def main():
    print(f"Starting ICR Processing in: {ICR_IMAGE_DIR}")
    
    if not os.path.isdir(ICR_IMAGE_DIR):
        print("❌ Base directory not found.")
        return

    # --- Step 1: Collect and Process ICR Images ---
    images_to_process, all_extracted_data = collect_icr_images(ICR_IMAGE_DIR)
    all_extracted_data = call_icr_api(images_to_process, all_extracted_data)
    save_json(all_extracted_data, ICR_OUTPUT_JSON)
    # clean_icr_json(ICR_OUTPUT_JSON)   # optional if needed

    # --- Step 2: Merge with Generalized JSON ---    
    merge_icr_fields_with_generalized_json(
        base_json_path=os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, f"{batch_name}.json"),
        icr_json_path=ICR_OUTPUT_JSON,
        output_path=os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, f"{batch_name}_1.json"),
        key_fields_json=os.path.join(base_folder, "Annotations", omr_template_name, "key_fields.json")
    )

    # --- Step 3: Merge ICR with ED Results JSON & CSV ---
    merge_icr_with_ed_results(
        icr_results=all_extracted_data,
        ed_json_path=ED_JSON_PATH,
        ed_csv_path=ED_CSV_PATH
    )

# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    main()