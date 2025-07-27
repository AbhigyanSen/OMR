import os
import json
import csv
import requests

# ---------------- Configuration ---------------- #
API_URL = "http://10.4.1.66:8003/predict"

base_folder = r"D:\Projects\OMR\new_abhigyan\Restructure"

batch_name = "BE24-05-07"
omr_template_name = "ASSAMOMR"
date = "23072025"

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

# Generate a generalized JSON file for the batch ---------------------------------------------------------
def merge_icr_fields_with_generalized_json(base_json_path, icr_json_path, output_path):
    # Load JSONs
    with open(base_json_path, "r") as f:
        gen_data = json.load(f)
    with open(icr_json_path, "r") as f:
        icr_data = json.load(f)

    for image in gen_data.get("IMAGES", []):
        img_name = os.path.splitext(os.path.basename(image["IMAGENAME"]))[0]
        icr_fields = icr_data.get(img_name, {})

        # ----- Find original roll_no and booklet_no fields -----
        roll_field = next((f for f in image["FIELDS"] if f["FIELD"] == "roll_no"), None)
        booklet_field = next((f for f in image["FIELDS"] if f["FIELD"] == "booklet_no"), None)

        # ----- ICR Roll -----
        if roll_field:
            icr_value = icr_fields.get("roll", "")
            success = "Y" if icr_value and "error" not in icr_value.lower() else "N"
            image["FIELDS"].append({
                "FIELD": "roll_no ICR",
                "XCORD": roll_field["XCORD"],
                "YCORD": roll_field["YCORD"],
                "WIDTH": roll_field["WIDTH"],
                "HEIGHT": roll_field["HEIGHT"],
                "FIELDDATA": icr_value,
                "CONFIDENCE": "X90",
                "SUCCESS": success
            })

        # ----- ICR Booklet -----
        if booklet_field:
            icr_value = icr_fields.get("booklet", "")
            success = "Y" if icr_value and "error" not in icr_value.lower() else "N"
            image["FIELDS"].append({
                "FIELD": "booklet_no ICR",
                "XCORD": booklet_field["XCORD"],
                "YCORD": booklet_field["YCORD"],
                "WIDTH": booklet_field["WIDTH"],
                "HEIGHT": booklet_field["HEIGHT"],
                "FIELDDATA": icr_value,
                "CONFIDENCE": "X80",
                "SUCCESS": success
            })

        # ---- Assign sequence numbers ----
        for idx, field in enumerate(image["FIELDS"], start=1):
            field["SEQUENCE"] = idx

    # Save merged JSON
    with open(output_path, "w") as f:
        json.dump(gen_data, f, indent=4)

    print(f"✅ Merged ICR fields added with sequence numbers and saved to {output_path}")


# ---------------- Main Orchestration ---------------- #

def main():
    print(f"Starting ICR Processing in: {ICR_IMAGE_DIR}")
    
    if not os.path.isdir(ICR_IMAGE_DIR):
        print("❌ Base directory not found.")
        return

    images_to_process, all_extracted_data = collect_icr_images(ICR_IMAGE_DIR)
    all_extracted_data = call_icr_api(images_to_process, all_extracted_data)
    save_json(all_extracted_data, ICR_OUTPUT_JSON)

    merge_icr_with_ed_results(
        icr_results=all_extracted_data,
        ed_json_path=ED_JSON_PATH,
        ed_csv_path=ED_CSV_PATH
    )
     
    merge_icr_fields_with_generalized_json(
        base_json_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, f"{batch_name}.json"),
        icr_json_path = ICR_OUTPUT_JSON, 
        output_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, f"{batch_name}.json")
        )


# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    main()