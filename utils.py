
import cv2
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

#
# def apply_blur(image_path, output_folder, kernel_size=(3, 3)):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Unable to load image from {image_path}")
#         return
#
#     # Apply Gaussian blur
#     blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
#
#     # Save the blurred image
#     filename = os.path.basename(image_path)
#     output_path = os.path.join(output_folder, f"blurred_{filename}")
#     cv2.imwrite(output_path, blurred_image)
#
#     print(f"Blurred image saved to {output_path}")
#
#
# def apply_resize(image_path, output_folder, target_size=(64, 64)):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Unable to load image from {image_path}")
#         return
#
#     # Resize the image
#     resized_image = cv2.resize(image, target_size)
#
#     # Save the resized image
#     filename = os.path.basename(image_path)
#     output_path = os.path.join(output_folder, f"resized_{filename}")
#     cv2.imwrite(output_path, resized_image)
#
#     print(f"Resized image saved to {output_path}")
#
#
# def augment_data(original_image_path, output_folder, num_augmented_images=5):
#     # Load the original image
#     original_image = Image.open(original_image_path)
#
#     # Define torchvision transforms for data augmentation
#     data_transforms = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(degrees=30),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #..
#         transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
#         transforms.ToTensor(),
#     ])
#
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Apply data augmentation and save augmented images
#     for i in range(num_augmented_images):
#         transformed_image = data_transforms(original_image)
#         augmented_image_path = os.path.join(output_folder, f"augmented_{i + 1}.jpg")
#         transforms.ToPILImage()(transformed_image).save(augmented_image_path)
#
#         print(f"Augmented image {i + 1} saved to {augmented_image_path}")
#
import cv2
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import random
def apply_blur(image_path, output_folder, kernel_size=(7,7)):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    #Apply median blur
    # blurred_image = cv2.medianBlur(image, kernel_size)

    # Save the blurred image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"blurred_{filename}")
    cv2.imwrite(output_path, blurred_image)

    print(f"Blurred image saved to {output_path}")

def generate_unique_random_numbers(upper_bound, count):
    if count > upper_bound:
        raise ValueError("Number of unique random numbers cannot exceed the range")

    return random.sample(range(upper_bound + 1), count)

def apply_resize(image_path, output_folder, target_size=(256,256)):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Resize the image
    resized_image = cv2.resize(image, target_size)

    # Save the resized image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"resized_{filename}")
    cv2.imwrite(output_path, resized_image)

    print(f"Resized image saved to {output_path}")

def augment_data(original_image_path, output_folder, face_coordinates, num_augmented_images=3, should_add_jitter=True, prefix= ""):
    # Load the original image
    original_image = Image.open(original_image_path)

    # Extract face region from the original image based on the provided face coordinates
    # left, top, right, bottom = face_coordinates
    # face_image = original_image.crop((left, top, right, bottom))

    # Convert face image to grayscale
    face_image_gray = original_image.convert('L')

    composition_steps= [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
    ]

    if not should_add_jitter:
        composition_steps = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
    ]
    # Define torchvision transforms for data augmentation
    data_transforms = transforms.Compose(composition_steps)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Apply data augmentation and save augmented images
    for i in range(num_augmented_images):
        # Apply different transformations to each augmented image
        transformed_image = data_transforms(face_image_gray)
        augmented_image_path = os.path.join(output_folder, f"{prefix}augmented_{i + 1}.jpg")
        transforms.ToPILImage()(transformed_image).save(augmented_image_path)

        print(f"Augmented image {i + 1} saved to {augmented_image_path}")

