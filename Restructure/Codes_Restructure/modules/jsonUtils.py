import json
import re
import os


def process_omr_data(json_file_path, key_fields_path, template_name, output_file_path):
    """
    Processes OMR data from a JSON file, validates against a template name,
    and saves the processed data to a new JSON file.

    Args:
        json_file_path (str): The full path to the input JSON file.
        template_name (str): The template name to match against.
        output_file_path (str): The full path where the processed JSON file will be saved.

    Returns:
        None, but prints status messages.
    """    
    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return
    
    try:
        with open(key_fields_path, 'r') as f:
            key_fields = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{key_fields_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{key_fields_path}' is not a valid JSON file.")
        return

    # Match the template name
    if json_data.get("TEMPLATE") != template_name:
        print(f"Error: Template name mismatch. Expected '{template_name}', but found '{json_data.get('TEMPLATE')}' from the file.")
        return

    # Create a deep copy to avoid modifying the original data directly
    processed_json_data = json.loads(json.dumps(json_data))

    # Create key fields list
    key_fields_list = []
    for key, value in key_fields.items():
        if isinstance(value, str):
            key_fields_list.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    key_fields_list.append(item)

    print("Info: Key fields list:", key_fields_list)
    
    key_fields_icr_list = []
    for i, key in enumerate(key_fields_list):
        key_fields_icr_list.append(key + " ICR")

    print("Info: Key fields list with 'ICR' suffix:", key_fields_icr_list)

    # Iterate through each image
    if "IMAGES" in processed_json_data and isinstance(processed_json_data["IMAGES"], list):
        for image in processed_json_data["IMAGES"]:
            # Iterate through each field in the image
            if "FIELDS" in image and isinstance(image["FIELDS"], list):
                for field in image["FIELDS"]:

                    # Check for ICR specific fields
                    if field.get("FIELD") in key_fields_icr_list:
                        field_data = field.get("FIELDDATA", "")
                        
                        # Process "REGISTRATION NUMBER ICR"
                        if field["FIELD"].upper() == "REGISTRATION NUMBER ICR":
                            # Remove leading/trailing spaces and keep only digits, ~ and *
                            cleaned_data = re.sub(r'[^0-9~*]', '', field_data.strip())
                            field["FIELDDATA"] = cleaned_data
                        
                        # Process "ROLL NUMBER ICR"
                        elif field["FIELD"].upper() == "ROLL NUMBER ICR":
                            # Remove leading/trailing spaces and keep only digits, ~ and *
                            cleaned_data = re.sub(r'[^0-9~*]', '', field_data.strip())
                            # Insert a space after the first 6 digits
                            if len(cleaned_data) > 6:
                                field["FIELDDATA"] = cleaned_data[:6] + " " + cleaned_data[6:]
                            else:
                                field["FIELDDATA"] = cleaned_data
                        
                        # Process "QUESTION BOOKLET NUMBER ICR"
                        elif field["FIELD"].upper() == "QUESTION BOOKLET NUMBER ICR":
                            # Remove leading/trailing spaces and keep only digits, ~ and *
                            cleaned_data = re.sub(r'[^0-9~*]', '', field_data.strip())
                            # Insert a hyphen after the first 2 digits
                            if len(cleaned_data) > 2:
                                field["FIELDDATA"] = cleaned_data[:2] + "-" + cleaned_data[2:]
                            else:
                                field["FIELDDATA"] = cleaned_data
                                
                    # Check for OMR specific fields
                    if field.get("FIELD") in key_fields_list:
                        field_data = field.get("FIELDDATA", "")
                        
                        # Process "REGISTRATION NUMBER ICR"
                        if field["FIELD"].upper() == "REGISTRATION NUMBER":
                            # Remove leading/trailing spaces and keep only digits, ~ and *
                            cleaned_data = re.sub(r'[^0-9~*]', '', field_data.strip())
                            field["FIELDDATA"] = cleaned_data
                        
                        # Process "ROLL NUMBER ICR"
                        elif field["FIELD"].upper() == "ROLL NUMBER":
                            # Remove leading/trailing spaces and keep only digits, ~ and *
                            cleaned_data = re.sub(r'[^0-9~*]', '', field_data.strip())
                            # Insert a space after the first 6 digits
                            if len(cleaned_data) > 6:
                                field["FIELDDATA"] = cleaned_data[:6] + " " + cleaned_data[6:]
                            else:
                                field["FIELDDATA"] = cleaned_data
                        
                        # Process "QUESTION BOOKLET NUMBER ICR"
                        elif field["FIELD"].upper() == "QUESTION BOOKLET NUMBER":
                            # Remove leading/trailing spaces and keep only digits, ~ and *
                            cleaned_data = re.sub(r'[^0-9~*]', '', field_data.strip())
                            # Insert a hyphen after the first 2 digits
                            if len(cleaned_data) > 2:
                                field["FIELDDATA"] = cleaned_data[:2] + "-" + cleaned_data[2:]
                            else:
                                field["FIELDDATA"] = cleaned_data

    # Create the output folder if it doesn't exist
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the modified JSON to the new file path
    with open(output_file_path, 'w') as f:
        json.dump(processed_json_data, f, indent=4)
    
    print(f"Output: Successfully processed data and saved the output to '{output_file_path}'.")
    return output_file_path


def process_key_coordinates(field_mappings_path, key_fields_path, output_path):
    """
    Extracts the top-left and bottom-right coordinates for key fields
    from field_mappings.json and saves them with their actual names
    to a new JSON file.

    Args:
        field_mappings_path (str): Path to the field_mappings.json file.
        key_fields_path (str): Path to the key_fields.json file.
        output_path (str): Path to save the new JSON file.
    """
    try:
        with open(field_mappings_path, 'r') as f:
            field_mappings_data = json.load(f)
        with open(key_fields_path, 'r') as f:
            key_fields = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: A file was not found. {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: A file is not a valid JSON. {e}")
        return

    results = {}

    # The field_mappings.json file has a single image as a top-level key.
    # We iterate over the items to get the image name and its data.
    for image_name, image_data in field_mappings_data.items():
        if "mapped_fields" in image_data:
            image_results = {}

            # Iterate through the main key fields (e.g., "key0", "key1")
            for key_name in key_fields.keys():
                
                top_left_coords = None
                bottom_right_coords = None

                # Get top-left coordinates from main key's bbox
                if key_name in image_data["mapped_fields"] and "bbox" in image_data["mapped_fields"][key_name]:
                    actual_name = key_fields[key_name]

                    top_left_coords = image_data["mapped_fields"][key_name]["bbox"][:2]

                # Get bottom-right from last bubble's bbox
                if top_left_coords:
                    last_bubble_key = None
                    max_n2 = -1
                    max_n3 = -1

                    # REGEX to match keyN1_N2_N3
                    pattern = re.compile(rf"^{key_name}_(\d+)_(\d+)$")

                    for field_key in image_data["mapped_fields"].keys():
                        match = pattern.match(field_key)
                        if match:
                            n2 = int(match.group(1))
                            n3 = int(match.group(2))
                            if n2 > max_n2 or (n2 == max_n2 and n3 > max_n3):
                                max_n2 = n2
                                max_n3 = n3
                                last_bubble_key = field_key

                    if last_bubble_key and last_bubble_key in image_data["mapped_fields"]:
                        bottom_right_coords = image_data["mapped_fields"][last_bubble_key].get("bbox", [])[2:]
                
                if top_left_coords is not None and bottom_right_coords is not None:
                    # image_results[actual_name] = {
                    image_results[key_name] = {
                        "top_left": top_left_coords,
                        "bottom_right": bottom_right_coords
                    }
            
            if image_results:
                results[image_name] = image_results

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Output: Successfully extracted key field coordinates and saved to '{output_path}'.")
    return output_path


def update_key_field_dimensions(output_json_path, key_field_coordinates_path, key_fields_path):
    """
    Calculates width and height for key fields from coordinate data and
    updates the corresponding values in the output JSON file.

    Args:
        output_json_path (str): Path to the main output JSON file.
        key_field_coordinates_path (str): Path to the JSON file with key field coordinates.
        key_fields_path (str): Path to the JSON file with key field mappings.
    """
    try:
        # Load the necessary JSON data
        with open(output_json_path, 'r') as f:
            output_data = json.load(f)
        with open(key_field_coordinates_path, 'r') as f:
            coordinates_data = json.load(f)
        with open(key_fields_path, 'r') as f:
            key_fields = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: A file was not found. {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: A file is not a valid JSON. {e}")
        return
    
    # Invert the key_fields dictionary for easy lookup (e.g., {"REGISTRATION NUMBER": "key0"})
    key_fields_inverted = {v: k for k, v in key_fields.items()}
    # print("Debug: Key Fields Inverted: ", key_fields_inverted)

    # Iterate through each image in the output data
    for image_data in output_data.get("IMAGES", []):
        image_name = os.path.basename(image_data.get("IMAGENAME", ""))
        # print("Debug: Image Name: ", image_name)

        # Check if we have coordinate data for this image
        if image_name in coordinates_data:
            image_coords = coordinates_data[image_name]

            # Iterate through each field in the image
            for field in image_data.get("FIELDS", []):
                field_name = field.get("FIELD")
                # print("Debug: Field Name: ", field_name)

                # Check if the field is a key field we need to update
                if field_name in key_fields_inverted:
                    key_name = key_fields_inverted[field_name]
                    # print("Debug: Key Name: ", key_name)

                    # Check if we have coordinates for this key field
                    if key_name in image_coords:
                        coords = image_coords[key_name]
                        top_left = coords.get("top_left")
                        bottom_right = coords.get("bottom_right")
                        
                        if top_left and bottom_right:
                            # Calculate new width and height
                            new_width = bottom_right[0] - top_left[0]
                            new_height = bottom_right[1] - top_left[1]

                            # Update the WIDTH and HEIGHT fields
                            field["WIDTH"] = str(new_width)
                            field["HEIGHT"] = str(new_height)
                            # print(f"Debug: Updated dimensions for '{field_name}' in '{image_name}': WIDTH='{new_width}', HEIGHT='{new_height}'")

    # Save the modified JSON back to the original file path
    try:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Output: Successfully updated key field dimensions and saved to '{output_json_path}'.")
    except Exception as e:
        print(f"Error: Could not save the updated file. {e}")
    
    return output_json_path


# def main():
#     """
#     Main function to handle user input and call the processing functions.
#     """
#     # template_choices = ["HSOMR", "ASSAMOMR"]
    
#     # print("Please select a template name:")
#     # for i, choice in enumerate(template_choices):
#     #     print(f"{i+1}. {choice}")
    
#     # # The interactive input section has been commented out to prevent execution errors.
#     # # A default choice is used for demonstration.
#     # while True:
#     #     try:
#     #         user_choice = int(input("Input: Enter the number corresponding to the template: "))
#     #         if 1 <= user_choice <= len(template_choices):
#     #             template_to_use = template_choices[user_choice - 1]
#     #             break
#     #         else:
#     #             print("Error: Invalid choice. Please enter a number from the list.")
#     #     except ValueError:
#     #         print("Error: Invalid input. Please enter a number.")
#     # template_to_use = "HSOMR"

#     omr_template_name = "HSOMR"  # Default template name for demonstration
            
#     # Take the full path to the json file and the output folder path
#     json_file_path = "D:/OMR_DEV/abhigyan/BATCH018 copy.json"
#     output_folder = "D:/OMR_DEV/abhigyan/final_output"
#     key_fields_path = "D:/OMR_DEV/abhigyan/key_fields.json"

#     # Define the output file path by getting just the filename from the full path
#     base_filename = os.path.basename(json_file_path)
#     output_file_path = os.path.join(output_folder, base_filename)
    
#     # # Call the main processing function
#     # process_omr_data(json_file_path, key_fields_path, omr_template_name, output_file_path)

#     # Take the full paths to the json files and the output folder path
#     field_mappings_path = "D:/OMR_DEV/abhigyan/field_mappings.json"
#     key_fields_path = "D:/OMR_DEV/abhigyan/key_fields.json"
#     output_folder = "D:/OMR_DEV/abhigyan/"
    
#     # Define the output file path
#     coordinates_output_path = os.path.join(output_folder, "key_field_coordinates.json")
    
#     # Call the main processing function
#     # process_key_coordinates(field_mappings_path, key_fields_path, coordinates_output_path)

#     output_file_path = process_omr_data(
#         json_file_path=json_file_path, 
#         key_fields_path=key_fields_path, 
#         template_name=omr_template_name, 
#         output_file_path=json_file_path)
#     print(f"Debug: Output File Path: {output_file_path}")

#     coordinates_output_path = process_key_coordinates(
#         field_mappings_path=field_mappings_path, 
#         key_fields_path=key_fields_path, 
#         output_path=coordinates_output_path)
#     print(f"Debug: Coordinates Output Path: {coordinates_output_path}")

#     key_fields_path = "D:/OMR_DEV/abhigyan/key_fields.json"

#     # NEW: Call the function to update the dimensions in the output JSON
#     path = update_key_field_dimensions(
#         output_json_path=output_file_path,
#         key_field_coordinates_path=coordinates_output_path,
#         key_fields_path=key_fields_path
#     )
#     print(f"Debug: Fully Updated Output File Path: {path}")


# if __name__ == "__main__":
#     main()


def json_restructure(base_folder, omr_template_name, date, batch_name):
    
    # Path setup for function calls
    output_json_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, f"{batch_name}.json")
    field_mappings_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, "annotate_" + batch_name, "field_mappings.json")
    key_fields_path = os.path.join(base_folder, "Annotations", omr_template_name, "key_fields.json")
    key_fields_coordinates_output_path = os.path.join(base_folder, "Images", omr_template_name, date, "Output", batch_name, "annotate_" + "batch_name", "key_fields_coordinates.json")
    
    # Function calls
    output_file_path = process_omr_data(
        json_file_path=output_json_path,
        key_fields_path=key_fields_path,
        template_name=omr_template_name,
        output_file_path=output_json_path) # Both INPUT and OUTPUT path are same so as to make the changes directly into the original file
    
    coordinates_output_path = process_key_coordinates(
        field_mappings_path=field_mappings_path,
        key_fields_path=key_fields_path,
        output_path=key_fields_coordinates_output_path) # OUTPUT is saved to the same folder as key_fields.json for easier access and error proofing

    updated_output_file_path = update_key_field_dimensions(
        output_json_path=output_file_path,
        key_field_coordinates_path=coordinates_output_path,
        key_fields_path=key_fields_path
    )