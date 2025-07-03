import cv2
import numpy as np
import os
import json

class OptionMapper:
    def __init__(self, image_path, annotations_path, classes_path, anchor_data):
        self.image_path = image_path
        self.annotations_path = annotations_path
        self.classes_path = classes_path
        
        self.anchor_data = anchor_data # Data for the current image from anchor_data.json
        
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
            # Ensure deskewed dimensions are integers for cv2.warpPerspective
            self.deskewed_width = int(self.deskewed_width)
            self.deskewed_height = int(self.deskewed_height)
            self.image = cv2.warpPerspective(self.original_image, self.M_transform, (self.deskewed_width, self.deskewed_height))
        else:
            self.image = self.original_image.copy()

        self.mapped_annotations = {} # To store final detected/deskewed coordinates

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
                    # Store normalized coordinates directly
                    norm_x_center = float(parts[1])
                    norm_y_center = float(parts[2])
                    norm_width = float(parts[3])
                    norm_height = float(parts[4])
                    annotations.append((self.classes[class_id], norm_x_center, norm_y_center, norm_width, norm_height))
        return annotations

    def map_and_draw(self):
        # Prepare the deskewed image for contour detection
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply adaptive thresholding for better bubble detection in varying light
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 4) # Adjust block size and C as needed

        for class_name, norm_x_center, norm_y_center, norm_width, norm_height in self.annotations:
            if "Anch" in class_name:
                continue # Skip anchors, they are handled by anchorDetection

            # Convert normalized original annotation to pixel coordinates on the original image
            original_x_center = norm_x_center * self.original_width
            original_y_center = norm_y_center * self.original_height
            original_width_px = norm_width * self.original_width
            original_height_px = norm_height * self.original_height

            original_x1 = original_x_center - original_width_px / 2
            original_y1 = original_y_center - original_height_px / 2
            original_x2 = original_x_center + original_width_px / 2
            original_y2 = original_y_center + original_height_px / 2

            # Transform original annotation coordinates to the deskewed image space
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
            
            # Use the transformed bbox as a *search region* for actual bubble detection
            search_bbox_x1, search_bbox_y1, search_bbox_x2, search_bbox_y2 = \
                int(transformed_x1), int(transformed_y1), int(transformed_x2), int(transformed_y2)

            # Define a buffer around the expected bubble location to search more broadly
            buffer_scale = 1.5 # Search 150% of the annotation box
            buffer_x = int((search_bbox_x2 - search_bbox_x1) * buffer_scale / 2)
            buffer_y = int((search_bbox_y2 - search_bbox_y1) * buffer_scale / 2)

            # Ensure minimum buffer to catch slightly shifted bubbles
            min_buffer = 15 # pixels
            buffer_x = max(buffer_x, min_buffer)
            buffer_y = max(buffer_y, min_buffer)


            search_x1 = max(0, search_bbox_x1 - buffer_x)
            search_y1 = max(0, search_bbox_y1 - buffer_y)
            search_x2 = min(self.deskewed_width, search_bbox_x2 + buffer_x)
            search_y2 = min(self.deskewed_height, search_bbox_y2 + buffer_y)

            # Validate search region
            if search_x2 <= search_x1 or search_y2 <= search_y1 or \
               search_x1 >= self.deskewed_width or search_y1 >= self.deskewed_height:
                print(f"Warning: Invalid search area for {class_name} after transformation. Skipping. Bbox: {search_bbox_x1, search_bbox_y1, search_bbox_x2, search_bbox_y2}")
                continue

            # Extract ROI from the thresholded image
            roi = thresh[search_y1:search_y2, search_x1:search_x2]

            if roi.shape[0] == 0 or roi.shape[1] == 0:
                print(f"Warning: ROI is empty for {class_name} (search area: {search_x1},{search_y1},{search_x2},{search_y2}). Skipping.")
                continue

            # Find contours within the ROI
            contours, _ = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            found_bubble_bbox = None
            max_contour_area = 0

            # Expected average dimensions of a bubble based on typical OMR form
            # These are relative to the *transformed* annotation dimensions, which are a good starting point.
            expected_bubble_width = (transformed_x2 - transformed_x1) 
            expected_bubble_height = (transformed_y2 - transformed_y1)
            
            # Define min/max area and aspect ratio for a valid bubble
            min_area = 0.5 * (expected_bubble_width * expected_bubble_height) # At least 50% of annotation area
            max_area = 2.0 * (expected_bubble_width * expected_bubble_height) # Max 200% of annotation area
            min_aspect_ratio = 0.7 # For circular/square bubbles
            max_aspect_ratio = 1.3


            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area or area > max_area: # Filter by size
                    continue

                (cx_roi, cy_roi, cw_roi, ch_roi) = cv2.boundingRect(c)
                aspect_ratio = cw_roi / float(ch_roi)

                if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio): # Filter by aspect ratio
                    continue
                
                # Further refine by circularity (if applicable, e.g., for bubbles)
                # If your bubbles are always circles, uncomment and adjust this:
                # perimeter = cv2.arcLength(c, True)
                # if perimeter == 0: continue
                # circularity = 4 * np.pi * area / (perimeter * perimeter)
                # if circularity < 0.6: continue # Adjust threshold (e.g., 0.6 for circles)

                # Take the largest contour that fits criteria as the bubble
                if area > max_contour_area:
                    max_contour_area = area
                    # Convert ROI-relative coordinates to full image coordinates
                    found_bubble_bbox = (search_x1 + cx_roi, search_y1 + cy_roi, 
                                         search_x1 + cx_roi + cw_roi, search_y1 + cy_roi + ch_roi)
            
            if found_bubble_bbox:
                new_x1, new_y1, new_x2, new_y2 = found_bubble_bbox

                # Ensure integer types for OpenCV drawing functions
                new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)

                # Clamp coordinates to be within image boundaries (again, just in case)
                img_h, img_w = self.image.shape[:2]
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(img_w - 1, new_x2)
                new_y2 = min(img_h - 1, new_y2)

                if new_x2 <= new_x1 or new_y2 <= new_y1:
                    print(f"Warning: Final bounding box for {class_name} became invalid after clamping. Skipping drawing.")
                    continue

                self.mapped_annotations[class_name] = (new_x1, new_y1, new_x2, new_y2)

                # Draw rectangle
                cv2.rectangle(self.image, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                
                # Draw text - ensure text position is also valid
                text_x = new_x1
                text_y = new_y1 - 5 # Put text slightly above the box
                if text_y < 0:
                    text_y = new_y1 + 15
                if text_x < 0:
                    text_x = 0
                if text_x > img_w - len(class_name) * 5: # Rough estimate to prevent text from going too far right
                    text_x = img_w - len(class_name) * 5
                
                cv2.putText(self.image, class_name, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            else:
                print(f"No suitable bubble contour found for {class_name} in search region {search_x1, search_y1, search_x2, search_y2}.")
                # If no bubble is found, you might want to fall back to the transformed annotation bbox
                # or simply not draw it. For now, we'll draw the transformed annotation as a fallback.
                
                # Fallback: Draw the transformed annotation bbox if no contour is found
                fallback_x1, fallback_y1, fallback_x2, fallback_y2 = \
                    int(transformed_x1), int(transformed_y1), int(transformed_x2), int(transformed_y2)

                # Clamp fallback coords too
                img_h, img_w = self.image.shape[:2]
                fallback_x1 = max(0, fallback_x1)
                fallback_y1 = max(0, fallback_y1)
                fallback_x2 = min(img_w - 1, fallback_x2)
                fallback_y2 = min(img_h - 1, fallback_y2)

                if fallback_x2 > fallback_x1 and fallback_y2 > fallback_y1: # Check validity
                    self.mapped_annotations[class_name] = (fallback_x1, fallback_y1, fallback_x2, fallback_y2)
                    cv2.rectangle(self.image, (fallback_x1, fallback_y1), (fallback_x2, fallback_y2), (0, 0, 255), 1) # Red, thinner line for fallback
                    
                    text_x = fallback_x1
                    text_y = fallback_y1 - 5
                    if text_y < 0: text_y = fallback_y1 + 15
                    if text_x < 0: text_x = 0
                    if text_x > img_w - len(class_name) * 5: text_x = img_w - len(class_name) * 5

                    cv2.putText(self.image, class_name + " (F)", (text_x, text_y), # (F) for Fallback
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return self.image, self.mapped_annotations

def process_folder(folder_path, annotations_file, classes_file, anchor_data_json_path):
    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    output_dir = f"annotate_{folder_name}"
    warning_dir = os.path.join(output_dir, "warnings")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(warning_dir, exist_ok=True)

    # Load all anchor data from the single JSON file
    try:
        with open(anchor_data_json_path, "r") as f:
            all_anchor_data = json.load(f)
        print(f"Successfully loaded anchor data from {anchor_data_json_path}")
    except FileNotFoundError:
        print(f"Error: Anchor data JSON file not found at {anchor_data_json_path}. Please ensure anchorDetection.py has run successfully.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {anchor_data_json_path}. Check file integrity.")
        return

    # Dictionary to store all mapped annotations for all images
    all_mapped_annotations_data = {} 

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing {image_path}...")

            # Get anchor data specific to this image
            image_specific_anchor_data = all_anchor_data.get(filename)

            if image_specific_anchor_data is None:
                print(f"Warning: Anchor data not found for {filename}. Skipping mapping for this image.")
                all_mapped_annotations_data[filename] = {"error": "Anchor data missing"}
                # Optionally, save original image to warnings if anchor data is critical
                cv2.imwrite(os.path.join(warning_dir, filename), cv2.imread(image_path))
                continue
            
            try:
                mapper = OptionMapper(image_path, annotations_file, classes_file, image_specific_anchor_data)
                mapped_image, mapped_annotations = mapper.map_and_draw() # Get mapped annotations

                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, mapped_image)
                print(f"Mapped and saved to {save_path}")

                all_mapped_annotations_data[filename] = mapped_annotations

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                warning_path = os.path.join(warning_dir, filename)
                # Attempt to save the original image to warnings if processing fails
                try:
                    original_img = cv2.imread(image_path)
                    if original_img is not None:
                        cv2.imwrite(warning_path, original_img)
                    print(f"Saved to warning folder: {warning_path}")
                except Exception as img_save_err:
                    print(f"Could not save original image to warning folder: {img_save_err}")
                
                # Record error in JSON data
                all_mapped_annotations_data[filename] = {"error": str(e)}
                
    # Save all mapped annotations to a JSON file
    mapped_annotations_json_path = os.path.join(output_dir, "mapped_annotations.json")
    with open(mapped_annotations_json_path, 'w') as f:
        json.dump(all_mapped_annotations_data, f, indent=2)
    print(f"All mapped annotations saved to {mapped_annotations_json_path}")


if __name__ == "__main__":
    folder_path = r"D:\Projects\OMR\new_abhigyan\Phase1\testData\BE23_Series"
    annotations_file = r"D:\Projects\OMR\new_abhigyan\Phase1\Options\BE23-01-01003.txt"
    classes_file = r"D:\Projects\OMR\new_abhigyan\Phase1\Options\classes.txt"
    anchor_data_json_path = r"D:\Projects\OMR\new_abhigyan\Phase2\anchor_BE23_Series\anchor_centers.json"

    process_folder(folder_path, annotations_file, classes_file, anchor_data_json_path)