import json

def generate_anchors():
    num_anchors = int(input("Enter the number of anchors: "))
    anchors = [f"anchor_{i+1}" for i in range(num_anchors)]
    print("\nAnchors:")
    print(anchors)
    return anchors


def generate_key_fields():
    num_keys = int(input("\nEnter the number of key fields: "))
    key_fields = [f"key{i}" for i in range(num_keys)]
    print("\nKey Fields:")
    print(key_fields)
    return key_fields


def get_key_field_names(key_fields):
    key_field_names = {}
    print("\n=== Key Field Naming ===")
    print("Now, let's assign meaningful names to your key fields.")
    for key in key_fields:
        name = input(f"Enter a name for {key}: ")
        key_field_names[key] = name
    print("\nKey Field Names:")
    for key, name in key_field_names.items():
        print(f"{key}: {name}")
    return key_field_names


def generate_key_digits(key_fields):
    key_digits = {}
    for key in key_fields:
        num_digits = int(input(f"\nEnter number of digits for {key}: "))
        digits = [f"{key}_{i}" for i in range(num_digits)]
        key_digits[key] = digits
    print("\nKey Digits:")
    for key, digits in key_digits.items():
        print(f"{key}: {digits}")
    return key_digits


def generate_key_digit_options(key_digits):
    key_digit_options = {}
    for key, digits in key_digits.items():
        options_per_digit = int(input(f"\nEnter number of options per digit for {key}: "))
        options = []
        for digit in digits:
            digit_options = [f"{digit}_{j}" for j in range(options_per_digit)]
            options.extend(digit_options)
        key_digit_options[key] = options
    print("\nKey Digit Options:")
    for key, opts in key_digit_options.items():
        print(f"{key}: {opts}")
    return key_digit_options


def generate_questions():
    num_questions = int(input("\nEnter the number of questions: "))
    questions = [f"question_{i}" for i in range(1, num_questions + 1)]
    print("\nQuestions:")
    print(questions)
    return questions


def generate_question_options(questions):
    num_options = int(input("\nEnter number of options per question (max 26): "))
    if num_options > 26:
        print("Maximum options supported is 26 (A-Z). Setting to 26.")
        num_options = 26
    option_labels = [chr(65 + i) for i in range(num_options)]  # ASCII A-Z
    all_options = {}

    for idx, question in enumerate(questions, start=1):  # start=1 to match question_1
        options = [f"{idx}{label}" for label in option_labels]
        all_options[question] = options

    print("\nQuestion Options:")
    for q, opts in all_options.items():
        print(f"{q}: {opts}")
    return all_options


def main():
    print("=== Anchor Generation ===")
    anchors = generate_anchors()

    print("\n=== Key Field Generation ===")
    key_fields = generate_key_fields()

    key_field_names = get_key_field_names(key_fields)

    print("\n=== Key Digits Generation ===")
    key_digits = generate_key_digits(key_fields)

    print("\n=== Key Digit Options Generation ===")
    key_digit_options = generate_key_digit_options(key_digits)

    print("\n=== Question Generation ===")
    questions = generate_questions()

    print("\n=== Question Options Generation ===")
    question_options = generate_question_options(questions)

    # Combine all labels for classes.txt
    all_labels = []
    all_labels.extend(anchors)
    all_labels.extend(key_fields) # The generic key labels (key0, key1, etc.)
    
    # Flatten key_digits values into the list
    for key, digits in key_digits.items():
        all_labels.extend(digits)
    
    # Flatten key_digit_options values into the list
    for key, options in key_digit_options.items():
        all_labels.extend(options)

    all_labels.extend(questions)
    
    # Flatten question_options values into the list
    for question, options in question_options.items():
        all_labels.extend(options)

    # Save all labels to classes.txt
    try:
        with open("classes.txt", "w") as f:
            for label in all_labels:
                f.write(label + "\n")
        print("\nAll generated labels saved to classes.txt")
    except IOError as e:
        print(f"Error saving labels to classes.txt: {e}")

    # Save key_field_names to classes.json
    try:
        with open("classes.json", "w") as f:
            json.dump(key_field_names, f, indent=4)
        print("Key field names saved to classes.json")
    except IOError as e:
        print(f"Error saving key field names to classes.json: {e}")


if __name__ == "__main__":
    main()