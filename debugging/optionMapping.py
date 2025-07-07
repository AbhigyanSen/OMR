# ONLY CHANGE IN MAP AND DRAW FUNCTION, REST ALL ARE SAME

import cv2
import numpy as np
import os
import json

class OptionMapper:
    def __init__(self, image_path, annotations_path, classes_path, anchor_data):
        self.image_path = image_path
        self.annotations_path = annotations_path
        self.classes_path = classes_path
        self.anchor_data = anchor_data

        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        self.original_height, self.original_width = self.original_image.shape[:2]
        self.classes = self._load_classes()
        self.annotations = self._load_annotations()

        self.M_transform = np.array(self.anchor_data.get("M_transform")) if self.anchor_data.get("M_transform") is not None else None
        self.deskewed_width = self.anchor_data.get("deskewed_width", self.original_width)
        self.deskewed_height = self.anchor_data.get("deskewed_height", self.original_height)

        if self.M_transform is not None:
            self.deskewed_width = int(self.deskewed_width)
            self.deskewed_height = int(self.deskewed_height)
            self.image = cv2.warpPerspective(self.original_image, self.M_transform, (self.deskewed_width, self.deskewed_height))
        else:
            self.image = self.original_image.copy()

        self.mapped_annotations = {}

    def _load_classes(self):
        with open(self.classes_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_annotations(self):
        annotations = []
        with open(self.annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    norm_x_center = float(parts[1])
                    norm_y_center = float(parts[2])
                    norm_width = float(parts[3])
                    norm_height = float(parts[4])
                    annotations.append((self.classes[class_id], norm_x_center, norm_y_center, norm_width, norm_height))
        return annotations

    def map_and_draw(self):
        anchors = self.anchor_data.get("anchors", {})
        anch1_x, anch1_y = anchors.get("Anch1", (None, None))

        for class_name, norm_x_center, norm_y_center, norm_width, norm_height in self.annotations:
            if "Anch" in class_name:
                continue  # skip anchor points

            original_x_center = norm_x_center * self.original_width
            original_y_center = norm_y_center * self.original_height
            original_width_px = norm_width * self.original_width
            original_height_px = norm_height * self.original_height

            original_x1 = original_x_center - original_width_px / 2
            original_y1 = original_y_center - original_height_px / 2
            original_x2 = original_x_center + original_width_px / 2
            original_y2 = original_y_center + original_height_px / 2

            transformed_x1, transformed_y1, transformed_x2, transformed_y2 = original_x1, original_y1, original_x2, original_y2

            if self.M_transform is not None:
                original_pts = np.float32([
                    [original_x1, original_y1], [original_x2, original_y1],
                    [original_x2, original_y2], [original_x1, original_y2]
                ]).reshape(-1, 1, 2)

                if np.isnan(original_pts).any() or np.isinf(original_pts).any():
                    print(f"⚠️ Skipping {class_name}: invalid original points (NaN or Inf)")
                    continue

                transformed_pts = cv2.perspectiveTransform(original_pts, self.M_transform).reshape(-1, 2)

                if np.isnan(transformed_pts).any() or np.isinf(transformed_pts).any():
                    print(f"⚠️ Skipping {class_name}: invalid transformed points (NaN or Inf)")
                    continue

                transformed_x1 = np.min(transformed_pts[:, 0])
                transformed_y1 = np.min(transformed_pts[:, 1])
                transformed_x2 = np.max(transformed_pts[:, 0])
                transformed_y2 = np.max(transformed_pts[:, 1])

            # ✅ Final safety check before drawing
            if None in [transformed_x1, transformed_y1, transformed_x2, transformed_y2]:
                print(f"⚠️ Skipping {class_name}: some transformed coordinates are None")
                continue

            if np.isnan([transformed_x1, transformed_y1, transformed_x2, transformed_y2]).any():
                print(f"⚠️ Skipping {class_name}: NaN values in transformed coordinates")
                continue

            # ✅ Convert to int for drawing
            x1 = int(transformed_x1)
            y1 = int(transformed_y1)
            x2 = int(transformed_x2)
            y2 = int(transformed_y2)

            # Save mapping info for downstream processing
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            delta_x = center_x - anch1_x if anch1_x is not None else None
            delta_y = center_y - anch1_y if anch1_y is not None else None

            self.mapped_annotations[class_name] = {
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
                "delta_from_Anch1": [delta_x, delta_y]
            }

            # ✅ Draw bounding box and label
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

        return self.image, self.mapped_annotations


def process_folder(folder_path, annotations_file, classes_file, anchor_data_json_path):
    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    output_dir = f"annotate_{folder_name}"
    warning_dir = os.path.join(output_dir, "warnings")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(warning_dir, exist_ok=True)

    try:
        with open(anchor_data_json_path, "r") as f:
            all_anchor_data = json.load(f)
    except Exception as e:
        print(f"Error loading anchor data: {e}")
        return

    all_mapped_annotations_data = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            # Skip if anchor data is missing or marked as invalid
            image_specific_anchor_data = all_anchor_data.get(filename)
            if (image_specific_anchor_data is None or 
                not image_specific_anchor_data.get("valid_for_option_mapping", False)):
                print(f"⛔ Skipping {filename}: invalid or missing anchor data.")
                continue

            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing {image_path}...")

            try:
                mapper = OptionMapper(image_path, annotations_file, classes_file, image_specific_anchor_data)
                mapped_image, mapped_annotations = mapper.map_and_draw()
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, mapped_image)
                mapped_annotations["valid_for_marked_option"] = True  # ✅ Mark as valid
                all_mapped_annotations_data[filename] = mapped_annotations
                print(f"✅ Saved mapped image: {save_path}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                try:
                    cv2.imwrite(os.path.join(warning_dir, filename), cv2.imread(image_path))
                except Exception as w:
                    print(f"⚠️ Failed to write warning image for {filename}: {w}")
                all_mapped_annotations_data[filename] = {
                    "error": str(e),
                    "valid_for_marked_option": False  # ❌ Mark as invalid for next stage
                }

    mapped_annotations_json_path = os.path.join(output_dir, "mapped_annotations.json")
    with open(mapped_annotations_json_path, 'w') as f:
        json.dump(all_mapped_annotations_data, f, indent=2)
    print(f"\n📝 All mapped annotations saved to {mapped_annotations_json_path}")


if __name__ == "__main__":
    folder_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series"
    annotations_file = r"D:\Projects\OMR\new_abhigyan\debugging\Annotations\detailed_annotations\Test_Series\TEST-01003.txt"
    classes_file = r"D:\Projects\OMR\new_abhigyan\debugging\Annotations\detailed_annotations\Test_Series\classes.txt"
    anchor_data_json_path = r"D:\Projects\OMR\new_abhigyan\debugging\anchor_Test_Series\anchor_centers.json"

    process_folder(folder_path, annotations_file, classes_file, anchor_data_json_path)