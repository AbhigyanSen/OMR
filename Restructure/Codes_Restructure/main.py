from modules.anchorDetection import process_batch
from modules.fieldMapping import process_field_mapping
from modules.markedOption import process_marked_options
from modules.runRequest import process_icr_requests
import sys, json

def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <omr_template_name> <date> <batch_name>")
        sys.exit(1)

    omr_template_name, date, batch_name = sys.argv[1:4]
    config = load_config()
    base_folder = config["base_folder"]

    # Anchor Detection for the batch
    results = process_batch(base_folder, omr_template_name, date, batch_name, save_anchor_images=False)
    print(f"|INFO| Batch processed. Images processed: {len(results)}")
    
    # Process field mapping for the batch
    field_mapping_results = process_field_mapping(base_folder, omr_template_name, date, batch_name, save_mapped_images=False)
    print(f"|INFO| Field mapping completed. Mappings found: {len(field_mapping_results)}")

    # Process marked options for the batch
    marked_stats = process_marked_options(base_folder, omr_template_name, date, batch_name, draw_bboxes=False)
    print(f"|INFO| Marked options processed. "
        f"Images: {marked_stats['processed_images']}, "
        f"Detected fields: {marked_stats['total_detected_fields']}")

    # Process ICR requests for the batch
    icr_stats = process_icr_requests(base_folder, omr_template_name, date, batch_name)
    print(f"|INFO| ICR processed images: {icr_stats['processed_images']}")