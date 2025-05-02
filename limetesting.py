import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Initialize the processor with the model identifier from Hugging Face's model hub

processor = ViTImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model_directory = r"model"
# Path to your model directory

# Load the model configuration and weights
model = ViTForImageClassification.from_pretrained(model_directory)

# Assuming you know the class names
class_names = ['angry', 'normal', 'tongue', 'happy']  # Update this with actual class names

# Function to predict with the model and return probabilities
def predict_with_model(images):
    inputs = processor(images=images, return_tensors="pt")

    # Predict with the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and convert to probabilities
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    return probs.detach().numpy()


# LIME explanation function
def explain_image(image_path, num_samples=300, top_labels=1):
    explainer = lime_image.LimeImageExplainer()

    # Load image using PIL
    image = Image.open(image_path).convert("RGB")

    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Explain instance
    explanation = explainer.explain_instance(image_np, predict_with_model,
                                             top_labels=top_labels, hide_color=0, num_samples=num_samples)

    return image, image_np, explanation  # Return original image, numpy array image, and explanation


# Function to visualize LIME explanations with the most important part only
def visualize_explanation(image_np, explanation, label):
    # Get weights for superpixels
    weights = explanation.local_exp[label]
    weights_dict = dict(weights)

    # Get image and mask
    temp, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=1, hide_rest=True  # Only show the top 1 feature
    )

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(image_np / 255.0, mask))
    plt.title('Most Important Part')

    plt.tight_layout()
    plt.show()


# Main loop to use LIME for explanations
while True:
    # Prompt for an image path
    image_path = input("Enter the path to your image, or type 'exit' to quit: ").strip()

    if image_path.lower() == 'exit':
        print("Exiting the program.")
        break

    try:
        original_image, image_np, explanation = explain_image(image_path)

        # Print LIME explanation results
        print("Top labels:", explanation.top_labels)

        # Iterate over the top labels and print details
        for label in explanation.top_labels:
            print(f"Explanation for label: {class_names[label]}")

            # Visualize explanations with only the most important part
            visualize_explanation(image_np, explanation, label)

    except Exception as e:
        print(f"An error occurred: {e}")
