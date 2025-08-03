import os
import json
import csv
import requests
import copy
import logging
from datetime import datetime

API_URL = "http://10.4.1.66:8003/predict"

# ---------- Logger Setup ----------
def setup_logger(batch_name):
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Single log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{batch_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger("RunRequest")
    logger.setLevel(logging.INFO)

    # File handler (no terminal logging)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s |%(levelname)s| %(message)s'))

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)

    return logger, log_path

# ---------- Helper Functions ----------
def collect_icr_images(base_dir, logger):
    all_extracted_data = {}
    images_to_process = []
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    logger.info(f"Detected categories: {categories}")

    for category in categories:
        category_path = os.path.join(base_dir, category)
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                full_path = os.path.join(category_path, filename)
                image_name = os.path.splitext(filename)[0]
                images_to_process.append((full_path, image_name, category))
                if image_name not in all_extracted_data:
                    all_extracted_data[image_name] = {cat: "" for cat in categories}
            else:
                logger.warning(f"Skipping non-image file: {filename}")
    return images_to_process, all_extracted_data

def call_icr_api(images_to_process, all_extracted_data, logger):
    for image_path, image_name, category in images_to_process:
        logger.info(f"Processing: {image_name} | Category: {category}")
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
                logger.info(f"Extracted: {image_name} | {category} -> {value}")
                if isinstance(value, str) and "error:" in value.lower():
                    all_extracted_data[image_name]["ERROR"] = _detect_error_code(value)
            else:
                msg = f"Error: {response.status_code}"
                all_extracted_data[image_name][category] = msg
                all_extracted_data[image_name]["ERROR"] = _detect_error_code(msg)
                logger.error(f"API failure ({image_name}): {msg}")
        except Exception as e:
            msg = f"Error: {str(e)}"
            all_extracted_data[image_name][category] = msg
            all_extracted_data[image_name]["ERROR"] = _detect_error_code(msg)
            logger.exception(f"Exception processing {image_name}: {e}")
    return all_extracted_data

def _detect_error_code(msg):
    lower_val = msg.lower()
    if "jsondecodeerror" in lower_val:
        return "1"
    elif "connectionabortederror" in lower_val:
        return "2"
    elif "httpconnectionpool" in lower_val:
        return "3"
    return "10"

def save_json(data, path, logger):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved ICR data to: {path}")

def merge_icr_fields_with_generalized_json(base_json_path, icr_json_path, output_path, key_fields_json, logger):
    with open(base_json_path, "r", encoding="utf-8") as f:
        gen_data = json.load(f)
    with open(icr_json_path, "r", encoding="utf-8") as f:
        icr_data = json.load(f)
    with open(key_fields_json, "r", encoding="utf-8") as f:
        key_fields = json.load(f)

    for image in gen_data.get("IMAGES", []):
        img_name = os.path.splitext(os.path.basename(image["IMAGENAME"]))[0]
        icr_fields = icr_data.get(img_name, {})
        new_field_list = []
        for field in image["FIELDS"]:
            # keep original
            new_field_list.append(copy.deepcopy(field))

            if field["FIELD"] in key_fields:
                icr_field_name = key_fields[field["FIELD"]]
                icr_value = icr_fields.get(icr_field_name, "")
                error_code = "" if not isinstance(icr_value, str) or "error:" not in icr_value.lower() else _detect_error_code(icr_value)
                if error_code:
                    icr_value = ""
                icr_field = copy.deepcopy(field)
                icr_field["FIELD"] = f"{field['FIELD']} ICR"
                icr_field["FIELDDATA"] = icr_value
                icr_field["CONFIDENCE"] = "100" if icr_value.strip() else ""
                icr_field["SUCCESS"] = "Y" if icr_value.strip() else "N"
                icr_field["ERRORICR"] = error_code
                new_field_list.append(icr_field)

        # ---- Convert field names to human-readable ----
        for f in new_field_list:
            field_name = f["FIELD"]

            # 1) From key_fields.json
            if field_name in key_fields:
                f["FIELD"] = key_fields[field_name]

            # 2) ICR variant of key field
            elif field_name.endswith(" ICR") and field_name[:-4] in key_fields:
                f["FIELD"] = key_fields[field_name[:-4]] + " ICR"

            # 3) Questions like question_1 → Q1
            elif field_name.startswith("question_") and field_name.split("_")[-1].isdigit():
                q_num = field_name.split("_")[-1]
                f["FIELD"] = f"Q{q_num}"

            # 5) Static OMR Sheet Number
            elif field_name == "omr_sheet_no":
                f["FIELD"] = "OMR Sheet Number"
                
            # Ensure ERRORICR present
            if "ERRORICR" not in f:
                f["ERRORICR"] = ""

        for idx, f in enumerate(new_field_list, start=1):
            f["SEQUENCE"] = idx

        image["FIELDS"] = new_field_list

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gen_data, f, indent=4)
    logger.info(f"ICR keys merged (with human-readable field names) and saved to {output_path}")
    
# Handling the OMR Sheet Number
def update_omr_sheet_number(base_json_path, icr_json_path, logger):
    # Load JSON files
    with open(base_json_path, "r", encoding="utf-8") as f:
        gen_data = json.load(f)
    with open(icr_json_path, "r", encoding="utf-8") as f:
        icr_data = json.load(f)

    for image in gen_data.get("IMAGES", []):
        img_name = os.path.splitext(os.path.basename(image["IMAGENAME"]))[0]
        omr_value = icr_data.get(img_name, {}).get("OMR Sheet Number", "")

        # Detect error code
        error_code = "" if not isinstance(omr_value, str) or "error:" not in omr_value.lower() else _detect_error_code(omr_value)
        if error_code:
            omr_value = ""

        # Find OMR Sheet Number field in generalized JSON
        for field in image["FIELDS"]:
            if field["FIELD"] == "OMR Sheet Number":
                field["FIELDDATA"] = omr_value
                field["CONFIDENCE"] = "100" if omr_value.strip() else ""
                field["SUCCESS"] = "Y" if omr_value.strip() else "N"
                field["ERRORICR"] = error_code
                break

    with open(base_json_path, "w", encoding="utf-8") as f:
        json.dump(gen_data, f, indent=4)
    logger.info(f"Updated OMR Sheet Number values from ICR into {base_json_path}")

def merge_icr_with_ed_results(icr_results, ed_json_path, ed_csv_path, logger):
    with open(ed_json_path, 'r', encoding='utf-8') as f:
        ed_data = json.load(f)
    for image_name, icr_fields in icr_results.items():
        key = f"{image_name}.jpg"
        if key in ed_data:
            for field, value in icr_fields.items():
                ed_data[key][f"ICR_{field}"] = value
        else:
            ed_data[key] = {f"ICR_{k}": v for k, v in icr_fields.items()}
    with open(ed_json_path.replace(".json", "_with_icr.json"), 'w', encoding='utf-8') as f:
        json.dump(ed_data, f, indent=4)
    logger.info(f"Updated ED JSON with ICR results: {ed_json_path.replace('.json', '_with_icr.json')}")

    with open(ed_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    fieldnames = reader[0].keys() if reader else []
    icr_fieldnames = {f"ICR_{field}" for v in icr_results.values() for field in v.keys()}
    new_fields = sorted(icr_fieldnames - set(fieldnames))
    final_fields = list(fieldnames) + new_fields
    updated_rows = []
    for row in reader:
        img_key = os.path.splitext(row.get("Image Name", ""))[0]
        icr_data = icr_results.get(img_key, {})
        for field, value in icr_data.items():
            row[f"ICR_{field}"] = value
        updated_rows.append(row)
    with open(ed_csv_path.replace(".csv", "_with_icr.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(updated_rows)
    logger.info(f"Updated ED CSV with ICR results: {ed_csv_path.replace('.csv', '_with_icr.csv')}")

# ---------- Main Entry ----------
def process_icr_requests(base_folder, omr_template_name, date, batch_name):
    logger, log_path = setup_logger(batch_name)

    icr_image_dir = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name,
                                 f"annotate_{batch_name}", "ICR")
    icr_output_json = os.path.join(icr_image_dir, "ICR_Images.json")
    ed_json_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name,
                                "options_" + batch_name, "ed_results_human.json")
    ed_csv_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name,
                               "options_" + batch_name, "ed_results_human.csv")
    generalized_json_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name,
                                         f"{batch_name}.json")
    key_fields_json = os.path.join(base_folder, "Annotations", omr_template_name, "key_fields.json")

    logger.info(f"Starting ICR Processing in: {icr_image_dir}")
    if not os.path.isdir(icr_image_dir):
        logger.error("ICR directory not found.")
        return {"processed_images": 0}

    images_to_process, all_extracted_data = collect_icr_images(icr_image_dir, logger)
    all_extracted_data = call_icr_api(images_to_process, all_extracted_data, logger)
    save_json(all_extracted_data, icr_output_json, logger)
    merge_icr_fields_with_generalized_json(
        base_json_path=generalized_json_path,
        icr_json_path=icr_output_json,
        output_path=generalized_json_path,
        key_fields_json=key_fields_json,
        logger=logger
    )
    
    update_omr_sheet_number(
        base_json_path=generalized_json_path,
        icr_json_path=icr_output_json,
        logger=logger
    )

    merge_icr_with_ed_results(all_extracted_data, ed_json_path, ed_csv_path, logger)
    logger.info(f"ICR processing complete. Total images processed: {len(images_to_process)}")

    return {"processed_images": len(images_to_process)}