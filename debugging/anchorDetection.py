import cv2
import numpy as np
import math
import os
import json
import csv

class OMRProcessor:
    def __init__(self, image_path, annotations_path, classes_path):
        self.image_path = image_path
        self.annotations_path = annotations_path
        self.classes_path = classes_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        self.original_image = self.image.copy()
        self.original_height, self.original_width, _ = self.image.shape
        
        # Load annotations and classes first, as they are needed for coordinates
        self.classes = self._load_classes() # Load classes.txt first
        self.annotations = self._load_annotations() # Annotations need original image dimensions
        
        print(f"Original Image dimensions: {self.original_width}x{self.original_height}")
        print(f"Loaded annotations: {self.annotations}")
        print(f"Loaded classes: {self.classes}")

        # Store the transformation matrix from original to deskewed image
        self.M_transform = None 
        # Store the deskewed image dimensions
        self.deskewed_width = self.original_width
        self.deskewed_height = self.original_height

    def _load_annotations(self):
        """
        Loads annotations from the Label Studio .txt file.
        The format is: class_id x_center y_center width height (normalized)
        Returns:
            dict: A dictionary mapping class IDs to a list of their bounding box coordinates.
        """
        annotations = {}
        try:
            with open(self.annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0]) 
                        # Convert normalized coordinates to pixel coordinates based on original image
                        x_center = float(parts[1]) * self.original_width
                        y_center = float(parts[2]) * self.original_height
                        norm_width = float(parts[3]) * self.original_width
                        norm_height = float(parts[4]) * self.original_height

                        # Calculate top-left and bottom-right corners
                        x1 = int(x_center - norm_width / 2)
                        y1 = int(y_center - norm_height / 2)
                        x2 = int(x_center + norm_width / 2)
                        y2 = int(y_center + norm_height / 2)

                        # Store as (x1, y1, x2, y2)
                        annotations.setdefault(class_id, []).append((x1, y1, x2, y2))
        except FileNotFoundError:
            print(f"Annotation file not found at {self.annotations_path}. Proceeding without it.")
        return annotations

    def _load_classes(self):
        """
        Loads class names from the classes.txt file.
        Returns:
            list: A list of class names.
        """
        classes = []
        try:
            with open(self.classes_path, 'r') as f:
                for line in f:
                    # Remove any common newline characters like \r and \n
                    classes.append(line.strip().replace('\r', '')) 
        except FileNotFoundError:
            print(f"Classes file not found at {self.classes_path}. Proceeding without it.")
        return classes

    def _get_class_id(self, class_name):
        """
        Gets the class ID for a given class name.
        Args:
            class_name (str): The name of the class.
        Returns:
            int: The ID of the class, or -1 if not found.
        """
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1

    def _find_omr_sheet_contour(self, img_to_process):
        """
        Detects the largest rectangular contour, assumed to be the OMR sheet.
        Args:
            img_to_process (np.array): The image to find the contour in.
        Returns:
            tuple: (contour, approx_poly) of the sheet, or (None, None) if not found.
        """
        gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and take the largest one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        sheet_contour = None
        approx_poly = None

        for contour in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True) # Looser epsilon for more robustness

            # If the approximated contour has 4 vertices, it's likely our sheet
            if len(approx) == 4:
                # Ensure the contour is sufficiently large (e.g., more than 50% of image area)
                # to filter out small noise or other rectangular elements. Adjust threshold as needed.
                if cv2.contourArea(contour) > (img_to_process.shape[0] * img_to_process.shape[1] * 0.4): # Reduced to 40%
                    sheet_contour = contour
                    approx_poly = approx
                    break
        
        if sheet_contour is None:
            print("Warning: Could not find a large 4-sided contour for the OMR sheet.")
        
        return sheet_contour, approx_poly

    def _order_points(self, pts):
        """
        Orders a list of 4 points (top-left, top-right, bottom-right, bottom-left).
        Args:
            pts (np.array): A 4x2 numpy array of points.
        Returns:
            np.array: Ordered points.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left has the smallest sum
        rect[2] = pts[np.argmax(s)]  # bottom-right has the largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # top-right has the smallest difference
        rect[3] = pts[np.argmax(diff)] # bottom-left has the largest difference
        return rect

    def _deskew_image(self, sheet_approx_poly):
        """
        Applies a perspective transformation to deskew the image based on the sheet's corners.
        This version attempts to keep the overall image content by using the original dimensions
        or a scaled version that fits the output.
        Args:
            sheet_approx_poly (np.array): Approximated polygon of the OMR sheet from the original image.
        Returns:
            np.array: Deskewed image.
        """
        if sheet_approx_poly is None:
            print("No sheet polygon provided for deskewing. Returning original image.")
            self.M_transform = None
            self.deskewed_width = self.original_width
            self.deskewed_height = self.original_height
            return self.original_image 

        pts = sheet_approx_poly.reshape(4, 2)
        ordered_pts = self._order_points(pts)

        # Calculate the dimensions of the deskewed image based on the ordered points
        (tl, tr, br, bl) = ordered_pts
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        
        # Ensure the output size is reasonable and not zero
        maxWidth = max(1, maxWidth)
        maxHeight = max(1, maxHeight)

        # Define the destination points for the perspective transform (a perfect rectangle)
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Get the perspective transformation matrix from original points to new rectangular points
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        
        # Apply the transformation to the original image
        deskewed_image = cv2.warpPerspective(self.original_image, M, (maxWidth, maxHeight))
        
        # Store the transformation matrix for later use to map annotations
        self.M_transform = M
        self.deskewed_width = maxWidth
        self.deskewed_height = maxHeight

        print(f"Deskewed image dimensions: {self.deskewed_width}x{self.deskewed_height}")
        return deskewed_image

    def detect_anchor_points(self):
        """
        Detects the anchor points (general shapes: circles, squares) in the OMR sheet.
        It first deskews the image, then searches for contours in transformed ROIs.
        Returns:
            tuple: (list of detected anchors, deskewed image, transformation matrix)
        """
        detected_anchors = []
        
        # Define the anchor class IDs. These map to the *index* in your classes.txt
        anchor_class_names = ['Anch1', 'Anch2', 'Anch3', 'Anch4']
        
        # Perform deskewing first. This will update self.M_transform and deskewed dimensions
        # We work on the original_image to find the contour for deskewing
        sheet_contour, sheet_approx_poly = self._find_omr_sheet_contour(self.original_image)
        
        # The result of deskewing will be assigned to `self.image`
        self.image = self._deskew_image(sheet_approx_poly)
        
        # Now `self.image` is the deskewed version (or original if no deskewing)
        # And `self.M_transform` holds the matrix from original_image to self.image

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Iterate through the expected anchor annotations (from original image coordinates)
        for class_name_str in anchor_class_names:
            class_id = self._get_class_id(class_name_str)

            if class_id != -1 and class_id in self.annotations:
                for original_bbox_coords in self.annotations[class_id]: # Use original annotations from the loaded file
                    x1_orig, y1_orig, x2_orig, y2_orig = original_bbox_coords

                    current_bbox_coords = (x1_orig, y1_orig, x2_orig, y2_orig) 

                    # If deskewing occurred, transform the original annotation coordinates
                    if self.M_transform is not None:
                        original_corners = np.float32([
                            [x1_orig, y1_orig], 
                            [x2_orig, y1_orig], 
                            [x2_orig, y2_orig], 
                            [x1_orig, y2_orig]  
                        ]).reshape(-1, 1, 2) 

                        transformed_corners = cv2.perspectiveTransform(original_corners, self.M_transform).reshape(-1, 2)
                        
                        x_min = int(np.min(transformed_corners[:, 0]))
                        y_min = int(np.min(transformed_corners[:, 1]))
                        x_max = int(np.max(transformed_corners[:, 0]))
                        y_max = int(np.max(transformed_corners[:, 1]))
                        
                        current_bbox_coords = (x_min, y_min, x_max, y_max)
                    
                    x1, y1, x2, y2 = current_bbox_coords # These are now the transformed (or original) bbox coords
                    
                    # Define a dynamic search area around the current bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Use a buffer that scales with the bbox size, but with a minimum
                    buffer_scale = 2.0 # Increased buffer for more robust search (200% buffer)
                    buffer_x = int(max(30, bbox_width * buffer_scale / 2)) # Increased min buffer
                    buffer_y = int(max(30, bbox_height * buffer_scale / 2)) 

                    search_x1 = max(0, x1 - buffer_x)
                    search_y1 = max(0, y1 - buffer_y)
                    search_x2 = min(self.deskewed_width, x2 + buffer_x)
                    search_y2 = min(self.deskewed_height, y2 + buffer_y)

                    if search_x2 <= search_x1 or search_y2 <= search_y1 or \
                       search_x1 >= self.deskewed_width or search_y1 >= self.deskewed_height:
                        print(f"Warning: Invalid search area for anchor {class_name_str} at {current_bbox_coords}. Skipping.")
                        continue

                    roi = blurred[search_y1:search_y2, search_x1:search_x2]
                    
                    if roi.shape[0] == 0 or roi.shape[1] == 0:
                        print(f"Warning: ROI is empty for anchor {class_name_str} at {current_bbox_coords}. Skipping.")
                        continue
                    
                    # --- General Shape Detection (replacing HoughCircles) ---
                    # Apply a binary threshold to the ROI
                    # Using Otsu's thresholding for automatic threshold calculation
                    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    # Find contours in the thresholded ROI
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    found_anchor_contour = None
                    min_area = 0.2 * bbox_width * bbox_height # Expected area 20% of bbox
                    max_area = 2.0 * bbox_width * bbox_height # Expected area 200% of bbox

                    # Filter contours to find the best candidate for an anchor
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area < min_area or area > max_area:
                            continue

                        # Get bounding box of the contour
                        (cx, cy, cw, ch) = cv2.boundingRect(c)
                        aspect_ratio = cw / float(ch)
                        
                        # Approximate the contour to a polygon to check number of vertices
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.04 * peri, True) # Adjusted epsilon for more tolerance
                        
                        # Check for square-like (4 vertices, aspect ratio close to 1) or circular (many vertices)
                        # For filled circles, approxPolyDP often gives many vertices or a somewhat circular shape
                        is_square = len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2
                        is_circle_like = len(approx) > 6 and 0.8 <= aspect_ratio <= 1.2 # Heuristic for circle (many vertices, near aspect 1)

                        if is_square or is_circle_like:
                            # Consider the largest valid contour as the anchor within the ROI
                            if found_anchor_contour is None or area > cv2.contourArea(found_anchor_contour):
                                found_anchor_contour = c
                    
                    if found_anchor_contour is not None:
                        (fx, fy, fw, fh) = cv2.boundingRect(found_anchor_contour)
                        
                        # Convert ROI coordinates back to full deskewed image coordinates
                        center_x = search_x1 + fx + fw // 2
                        center_y = search_y1 + fy + fh // 2
                        
                        det_x1 = search_x1 + fx
                        det_y1 = search_y1 + fy
                        det_x2 = search_x1 + fx + fw
                        det_y2 = search_y1 + fy + fh

                        detected_anchors.append({
                            'class_name': class_name_str,
                            'bbox': (det_x1, det_y1, det_x2, det_y2), # Bbox from actual detection on deskewed image
                            'center': (center_x, center_y),
                            'area': cv2.contourArea(found_anchor_contour),
                        })
                        print(f"Detected {class_name_str}: Center=({center_x}, {center_y}), BBox=({det_x1},{det_y1},{det_x2},{det_y2}), Area={cv2.contourArea(found_anchor_contour):.2f}")
                    else:
                        print(f"No suitable anchor contour detected in ROI for {class_name_str} (search bbox: {current_bbox_coords}, ROI size: {roi.shape[1]}x{roi.shape[0]})")
            else:
                print(f"Warning: Class {class_name_str} (ID: {class_id}) not found in annotations for detection.")

        return detected_anchors, self.image, self.M_transform # Return deskewed image and M_transform

    def visualize_results(self, detected_anchors, output_filename):
        """
        Draws bounding boxes and center points on the deskewed image and saves it.
        Args:
            detected_anchors (list): List of detected anchor dictionaries.
            output_filename (str): Name of the file to save the visualized image.
        Returns:
            dict: A dictionary mapping class names to their detected center points.
        """
        display_image = self.image.copy() # Use the deskewed image
        anchor_centers_for_json = {} # Dictionary to store class_name: [cx, cy]

        for anchor in detected_anchors:
            x1, y1, x2, y2 = anchor['bbox']
            center_x, center_y = int(anchor['center'][0]), int(anchor['center'][1])
            class_name = anchor['class_name']

            anchor_centers_for_json[class_name] = [center_x, center_y] # Store for JSON

            # Draw bounding box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw center point
            cv2.circle(display_image, (center_x, center_y), 5, (0, 0, 255), -1)
            # Put text label
            cv2.putText(display_image, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # Changed text color for visibility

        cv2.imwrite(output_filename, display_image)
        print(f"Results saved to {output_filename}")
        return anchor_centers_for_json # Return the dictionary for JSON saving


# Main execution
if __name__ == "__main__":

    # Define paths
    folder_path = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series"
    # If the layout (and thus anchor positions) changes per image, you might need dynamic annotation loading.
    annotations_file = r"D:\Projects\OMR\new_abhigyan\debugging\Annotations\anchor_annotations\Test_Series\labels\BLANK001.txt"  
    classes_file = r"D:\Projects\OMR\new_abhigyan\debugging\Annotations\anchor_annotations\Test_Series\classes.txt"

    # Create output directory based on folder name
    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    output_dir = os.path.join("anchor_" + folder_name)
    warning_dir = os.path.join(output_dir, "warnings")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(warning_dir, exist_ok=True)

    # CSV Path
    csv_output_path = os.path.join(output_dir, "anchor_centers.csv")
    
    # Initialize a dictionary to hold anchor data for all images
    anchor_json_path = os.path.join(output_dir, "anchor_centers.json") # New JSON file name
    all_image_anchor_data = {}

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing {image_path}...")

            try:
                processor = OMRProcessor(image_path, annotations_file, classes_file)
                detected_anchors, deskewed_img_result, M_transform_result = processor.detect_anchor_points()

                if len(detected_anchors) < 4:
                    print(f"⚠️ Not all 4 anchors detected for {filename}. Moving to warnings folder.")
                    warning_path = os.path.join(warning_dir, filename)
                    cv2.imwrite(warning_path, deskewed_img_result if deskewed_img_result is not None else processor.original_image)

                    all_image_anchor_data[filename] = {
                        "anchors": {anchor['class_name']: anchor['center'] for anchor in detected_anchors},
                        "M_transform": M_transform_result.tolist() if M_transform_result is not None else None,
                        "deskewed_width": processor.deskewed_width,
                        "deskewed_height": processor.deskewed_height,
                        "valid_for_option_mapping": False
                    }
                    continue
                
                else:
                    output_image_path = os.path.join(output_dir, filename)
                    anchor_centers_for_json = processor.visualize_results(detected_anchors, output_image_path)

                    print("\n--- Detected Anchor Details ---")
                    for anchor in detected_anchors:
                        print(f"Class: {anchor['class_name']}")
                        print(f"  Bounding Box (x1, y1, x2, y2): {anchor['bbox']}")
                        print(f"  Center (x, y): {anchor['center']}")
                        print(f"  Area: {anchor['area']:.2f}")
                        print("-" * 30)

                    all_image_anchor_data[filename] = {
                        "anchors": anchor_centers_for_json,
                        "M_transform": M_transform_result.tolist() if M_transform_result is not None else None,
                        "deskewed_width": processor.deskewed_width,
                        "deskewed_height": processor.deskewed_height,
                        "valid_for_option_mapping": True
                    }

                    print(f"Successfully processed {filename}. Anchor data stored.")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                try:
                    error_img = cv2.imread(image_path)
                    if error_img is not None:
                        warning_path = os.path.join(warning_dir, filename)
                        cv2.imwrite(warning_path, error_img)
                    all_image_anchor_data[filename] = {
                        "anchors": {},
                        "M_transform": None,
                        "deskewed_width": None,
                        "deskewed_height": None,
                        "valid_for_option_mapping": False,
                        "error": str(e)
                    }
                except Exception as img_err:
                    print(f"⚠️ Could not save error image: {img_err}")


    # Save all collected anchor data to a single JSON file
    with open(anchor_json_path, 'w') as f:
        json.dump(all_image_anchor_data, f, indent=2)
    print(f"\nAll anchor centers and transformation data saved to {anchor_json_path}")
    
    

    # Define the path to save the CSV
    # csv_output_path = os.path.join(output_dir, "anchor_centers.csv")

    # Define CSV column headers
    csv_headers = ["image_name", "Anch1", "Anch2", "Anch3", "Anch4"]

    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        for image_name, data in all_image_anchor_data.items():
            anchors = data.get("anchors", {})
            row = {
                "image_name": image_name,
                "Anch1": anchors.get("Anch1"),
                "Anch2": anchors.get("Anch2"),
                "Anch3": anchors.get("Anch3"),
                "Anch4": anchors.get("Anch4")
            }
            writer.writerow(row)

    print(f"Anchor centers also saved to CSV: {csv_output_path}")
