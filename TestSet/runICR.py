import os
import json
import tempfile
import uuid # Used for generating unique filenames for temporary files
from PIL import Image
import torch

from qwen_vl_utils import process_vision_info

# Hugging Face libraries for local model loading
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

# Flask libraries for creating the web API
from flask import Flask, request, jsonify

# --- Global variables for Qwen-VL model and processor ---
# These will hold the loaded model and processor to avoid reloading on every request.
_model = None
_processor = None
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# --- Utility function: process_vision_info (directly from your original context) ---
# This function prepares image inputs for the model. It's included here to keep
# the API file self-contained without needing an external qwen_vl_utils.py.
#def process_vision_info(messages):
#    """
#    Processes the vision information from the messages payload, extracting images.
#    """
#    images = []
#    videos = [] # Your original code included 'videos' but didn't explicitly process them for VL tasks.
#    for message in messages:
#        if message["role"] == "user":
#            for part in message["content"]:
#                if part["type"] == "image":
#                    images.append(part["image"])
#    return images, videos

# --- Model Loading Function (from your original code) ---
# This function handles the loading of the large Qwen-VL model into memory.
# It uses 4-bit quantization (BitsAndBytesConfig) to reduce memory usage,
# and attempts to load the model onto a GPU if available.
def _load_qwen_model_and_processor():
    """Loads the Qwen-VL model and processor using GPU if available."""
    global _model, _processor

    # If the model and processor are already loaded, we don't need to load them again.
    if _model is not None and _processor is not None:
        print("[INFO] Model and processor already loaded.")
        return

    # Configuration for 4-bit quantization, which helps in loading larger models
    # on GPUs with limited memory.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # Use float16 for computation if on GPU
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4" # NormalFloat4 quantization
    )

    print(f"[INFO] Loading model: {MODEL_NAME}")
    # Determine if CUDA (GPU) is available; otherwise, fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 on GPU for efficiency, float32 on CPU.
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        # Load the processor (tokenizer and image preprocessor)
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        # Load the vision-to-sequence model with specified dtype, device, and quantization.
        _model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else "cpu", # Automatically map model layers to available devices
            quantization_config=bnb_config
        )
        _model.eval() # Set the model to evaluation mode (disables dropout, etc.)
        print(f"[INFO] Model loaded successfully on {device.upper()} using dtype: {dtype}.")
    except Exception as e:
        print(f"[ERROR] Failed to load model {MODEL_NAME}: {e}")
        # Clear model and processor on failure to indicate an issue.
        _model = None
        _processor = None
        # Re-raise the exception to stop the application startup if model loading fails.
        raise RuntimeError(f"Failed to load Qwen-VL model: {e}")

# --- Qwen-VL Local Inference Function (from your original code) ---
# This function performs the core task: taking an image path and a category,
# and using the loaded Qwen-VL model to extract the requested handwritten data.
def qwen_vl_local(image_path: str, category: str):
    """
    Performs local inference using the Qwen-VL-3B-Instruct model
    to extract handwritten data for a specified category from an image.
    """
    # Ensure the model and processor are loaded before attempting inference.
    # This check acts as a fallback if _load_qwen_model_and_processor wasn't called
    # during app startup or failed silently for some reason.
    if _model is None or _processor is None:
        try:
            _load_qwen_model_and_processor()
        except RuntimeError:
            return {category: "Error: Model loading failed"}

    # Read and prepare the image from the provided path.
    print(f"[INFO] Loading image from: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB") # Ensure image is in RGB format
        print("[INFO] Image loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR]: Image file not found at {image_path}")
        return {category: "Error: Image file not found"}
    except Exception as e:
        print(f"[ERROR]: An error occurred while opening the image: {e}")
        return {category: f"Error: Image processing failed - {str(e)}"}

    # Construct the message payload for the chat completion.
    # This mimics a conversation where the user provides an image and a text prompt.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"""
                    Give me ONLY the handwritten data in the format: {{"{category}":"<handwritten data or None>"}}
                    """
                }
            ]
        }
    ]

    try:
        # Process vision inputs (images from messages).
        image_inputs, video_inputs = process_vision_info(messages)

        # Apply the chat template to format the text prompt according to the model's
        # expected input format.
        text_for_model = _processor.apply_chat_template(
            messages,
            tokenize=False, # We'll tokenize later with the full inputs
            add_generation_prompt=True # Add special tokens to start generation
        )

        # Prepare final model inputs (text and image features).
        # '.to(_model.device)' ensures inputs are on the same device (GPU/CPU) as the model.
        inputs = _processor(
            text=[text_for_model],
            images=image_inputs,
            videos=video_inputs,
            padding=False, # No padding for single inference; padding=True is for batching
            return_tensors="pt" # Return PyTorch tensors
        ).to(_model.device)

        # Generate the response from the model.
        print("[INFO] Generating response.")
        with torch.no_grad(): # Disable gradient calculation for inference to save memory
            if torch.cuda.is_available():
                # Use automatic mixed precision (autocast) on CUDA for faster inference
                # and reduced memory usage without significant loss of accuracy.
                with torch.cuda.amp.autocast():
                    output_tokens = _model.generate(
                        **inputs,
                        max_new_tokens=64, # Maximum length of the generated output
                        do_sample=False, # Use greedy decoding (no sampling) for consistent output
                        temperature=0.7, # Controls randomness, ignored if do_sample=False
                        top_p=0.9, # Controls token sampling, ignored if do_sample=False
                        pad_token_id=_processor.tokenizer.eos_token_id # Stop generation when EOS token is produced
                    )
            else:
                # Fallback for CPU (or if autocast is not desired/available)
                output_tokens = _model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=_processor.tokenizer.eos_token_id
                )

        # Decode the generated token IDs back into readable text.
        # We slice output_tokens to remove the input prompt part.
        generated_text = _processor.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print("[INFO] Raw model output (before JSON parsing):")
        print(generated_text)

        # The model might wrap JSON output in markdown code blocks (e.g., "```json\n...\n```").
        # This logic strips those delimiters to get a clean JSON string.
        json_output_string = generated_text
        if json_output_string.startswith("```json") and json_output_string.endswith("```"):
            json_output_string = json_output_string[len("```json"): -len("```")].strip()
            print("[INFO] Stripped markdown code block delimiters from the output.")
        elif json_output_string.startswith("```") and json_output_string.endswith("```"):
            json_output_string = json_output_string[len("```"): -len("```")].strip()
            print("[INFO] Stripped generic markdown code block delimiters from the output.")

        # Attempt to parse the cleaned string as JSON.
        parsed_json = json.loads(json_output_string)

        # Validate that the parsed JSON has the expected format: {"category": "data"}.
        if category in parsed_json and len(parsed_json) == 1:
            print("[OUTPUT] Parsed JSON object:")
            print(json.dumps(parsed_json, indent=2))
            return parsed_json
        else:
            print(f"[ERROR] Model output format unexpected. Expected {{\"'{category}'\":\"<data>\"}}, got: {parsed_json}")
            return {category: "Error: Unexpected format"}

    except json.JSONDecodeError as e:
        print(f"[ERROR] Decoding JSON: {e}")
        print(f"[ERROR] Raw output received: {generated_text}")
        return {category: "Error: JSONDecodeError"}
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return {category: f"Error: {str(e)}"}

# --- Flask Application Setup ---
app = Flask(__name__)

# Define the API endpoint that will receive requests.
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive an image file and a category string,
    then use Qwen-VL to extract handwritten data.
    """
    # 1. Input Validation: Check if both 'image' file and 'category' text are provided.
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400 # 400 Bad Request
    if 'category' not in request.form:
        return jsonify({"error": "No category provided"}), 400

    uploaded_file = request.files['image']
    category = request.form['category']

    if uploaded_file.filename == '':
        return jsonify({"error": "No selected image file"}), 400

    # 2. Save Temporary Image:
    # We save the uploaded image to a temporary file on the server. This allows
    # our qwen_vl_local function (which expects a file path) to process it.
    temp_dir = tempfile.gettempdir() # Get the system's temporary directory
    # Create a unique filename to avoid conflicts if multiple requests come in quickly.
    filename = f"{uuid.uuid4()}_{uploaded_file.filename}"
    temp_image_path = os.path.join(temp_dir, filename)

    try:
        uploaded_file.save(temp_image_path)
        print(f"[INFO] Image saved temporarily to: {temp_image_path}")

        # 3. Perform Inference: Call the Qwen-VL inference function.
        result = qwen_vl_local(image_path=temp_image_path, category=category)

        # 4. Return Response:
        # Ensure the result is a dictionary to be properly converted to JSON.
        if not isinstance(result, dict):
            result = {category: "Error: Inference returned unexpected type"}

        return jsonify(result), 200 # Return the result as JSON with a 200 OK status

    except Exception as e:
        # General error handling for any unexpected issues during the request processing.
        print(f"[ERROR] An error occurred during prediction: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500 # 500 Internal Server Error
    finally:
        # 5. Clean Up: Always remove the temporary image file, regardless of success or failure.
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print(f"[INFO] Cleaned up temporary file: {temp_image_path}")

# --- Main execution block ---
if __name__ == '__main__':
    # This block runs when the script is executed directly.
    # 1. Load the Model: It's CRITICAL to load the model ONLY ONCE when the application starts.
    # This prevents reloading the large model for every API request, saving time and memory.
    try:
        _load_qwen_model_and_processor()
    except RuntimeError as e:
        print(f"[CRITICAL ERROR] Application cannot start because the Qwen-VL model failed to load: {e}")
        # If the model can't load, the API won't function, so we exit.
        exit(1)

    # 2. Run the Flask App: Start the web server.
    # `debug=True` is useful for development (provides detailed errors, auto-reloads).
    # For production, set `debug=False` and use a production-ready WSGI server like Gunicorn.
    # `host='0.0.0.0'` makes the server accessible from any IP address (important for external access).
    # `port=5000` is the default Flask port.
    print("\n[INFO] Qwen-VL API server started.")
    app.run(debug=False, host='0.0.0.0', port=8003)
    print("[INFO] Qwen-VL API server stopped.")