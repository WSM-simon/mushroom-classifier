import argparse
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import keras
import matplotlib.pyplot as plt


def load_class_names(data_dir: str) -> List[str]:
    return sorted(
        [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
    )


def preprocess_image(image_path: str, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    print("Your image is: ")
    plt.imshow(img)
    plt.show()
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_top_k(predictions: np.ndarray, class_names: List[str], k: int = 3):
    probs = predictions[0]
    top_indices = np.argsort(probs)[-k:][::-1]
    return [(class_names[i], float(probs[i])) for i in top_indices]


def main():
    parser = argparse.ArgumentParser(description="Predict mushroom type from an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument(
        "--model",
        default="mushroom_model.keras",
        help="Path to the .keras model file",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to dataset root for class names",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    class_names = load_class_names(args.data_dir)
    Model = keras.models.load_model(args.model)

    img_array = preprocess_image(args.image)
    predictions = Model.predict(img_array)
    top_k = get_top_k(predictions, class_names, k=args.top_k)

    print("Top predictions:")
    for name, prob in top_k:
        print(f"{name}: {prob:.2%}")


if __name__ == "__main__":
    main()
