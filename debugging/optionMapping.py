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
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 4)

        anchors = self.anchor_data.get("anchors", {})
        anch1_x, anch1_y = anchors.get("Anch1", (None, None))

        for class_name, norm_x_center, norm_y_center, norm_width, norm_height in self.annotations:
            if "Anch" in class_name:
                continue

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
                    print(f"Warning: NaN or Inf in original_pts for {class_name}. Skipping transformation.")
                    continue

                transformed_pts = cv2.perspectiveTransform(original_pts, self.M_transform).reshape(-1, 2)

                if np.isnan(transformed_pts).any() or np.isinf(transformed_pts).any():
                    print(f"Warning: NaN or Inf in transformed_pts for {class_name}. Skipping.")
                    continue

                transformed_x1 = int(np.min(transformed_pts[:, 0]))
                transformed_y1 = int(np.min(transformed_pts[:, 1]))
                transformed_x2 = int(np.max(transformed_pts[:, 0]))
                transformed_y2 = int(np.max(transformed_pts[:, 1]))

            search_bbox_x1, search_bbox_y1, search_bbox_x2, search_bbox_y2 = \
                int(transformed_x1), int(transformed_y1), int(transformed_x2), int(transformed_y2)

            buffer_scale = 1.5
            buffer_x = max(int((search_bbox_x2 - search_bbox_x1) * buffer_scale / 2), 15)
            buffer_y = max(int((search_bbox_y2 - search_bbox_y1) * buffer_scale / 2), 15)

            search_x1 = max(0, search_bbox_x1 - buffer_x)
            search_y1 = max(0, search_bbox_y1 - buffer_y)
            search_x2 = min(self.deskewed_width, search_bbox_x2 + buffer_x)
            search_y2 = min(self.deskewed_height, search_bbox_y2 + buffer_y)

            if search_x2 <= search_x1 or search_y2 <= search_y1:
                print(f"Warning: Invalid search area for {class_name} after transformation. Skipping.")
                continue

            roi = thresh[search_y1:search_y2, search_x1:search_x2]
            if roi.size == 0:
                continue

            contours, _ = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            found_bubble_bbox = None
            max_contour_area = 0

            expected_bubble_width = (transformed_x2 - transformed_x1)
            expected_bubble_height = (transformed_y2 - transformed_y1)
            min_area = 0.5 * (expected_bubble_width * expected_bubble_height)
            max_area = 2.0 * (expected_bubble_width * expected_bubble_height)
            min_aspect_ratio = 0.7
            max_aspect_ratio = 1.3

            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area or area > max_area:
                    continue

                (cx_roi, cy_roi, cw_roi, ch_roi) = cv2.boundingRect(c)
                aspect_ratio = cw_roi / float(ch_roi)

                if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                    continue

                if area > max_contour_area:
                    max_contour_area = area
                    found_bubble_bbox = (search_x1 + cx_roi, search_y1 + cy_roi, 
                                         search_x1 + cx_roi + cw_roi, search_y1 + cy_roi + ch_roi)

            if found_bubble_bbox:
                new_x1, new_y1, new_x2, new_y2 = map(int, found_bubble_bbox)
                bubble_center_x = (new_x1 + new_x2) // 2
                bubble_center_y = (new_y1 + new_y2) // 2
                delta_x = bubble_center_x - anch1_x if anch1_x is not None else None
                delta_y = bubble_center_y - anch1_y if anch1_y is not None else None

                self.mapped_annotations[class_name] = {
                    "bbox": [new_x1, new_y1, new_x2, new_y2],
                    "center": [bubble_center_x, bubble_center_y],
                    "delta_from_Anch1": [delta_x, delta_y]
                }

                cv2.rectangle(self.image, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                cv2.putText(self.image, class_name, (new_x1, new_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            else:
                fallback_x1, fallback_y1, fallback_x2, fallback_y2 = map(int, (transformed_x1, transformed_y1, transformed_x2, transformed_y2))
                fallback_center_x = (fallback_x1 + fallback_x2) // 2
                fallback_center_y = (fallback_y1 + fallback_y2) // 2
                delta_x = fallback_center_x - anch1_x if anch1_x is not None else None
                delta_y = fallback_center_y - anch1_y if anch1_y is not None else None

                self.mapped_annotations[class_name] = {
                    "bbox": [fallback_x1, fallback_y1, fallback_x2, fallback_y2],
                    "center": [fallback_center_x, fallback_center_y],
                    "delta_from_Anch1": [delta_x, delta_y]
                }

                cv2.rectangle(self.image, (fallback_x1, fallback_y1), (fallback_x2, fallback_y2), (0, 0, 255), 1)
                cv2.putText(self.image, class_name + " (F)", (fallback_x1, fallback_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

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