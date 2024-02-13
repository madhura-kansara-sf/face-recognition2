import face_recognition
import cv2
import numpy as np

from utils import apply_blur, generate_unique_random_numbers
from utils import apply_resize
from utils import augment_data

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# # Load a sample picture and learn how to recognize it.
# alia_image = face_recognition.load_image_file("images/Alia_Bhatt2.jpg")
# alia_face_encoding = face_recognition.face_encodings(alia_image)[0]
#
# # Load a second sample picture and learn how to recognize it.
# m_image = face_recognition.load_image_file("images/madhura2 copy.jpg")
# m_face_encoding = face_recognition.face_encodings(m_image)[0]

# # Create arrays of known face encodings and their names
# known_face_encodings = [
#     alia_face_encoding,
#     m_face_encoding
# ]
# known_face_names = [
#     "Alia Bhatt",
#     "Madhura"
# ]


#load image folder
import os
from PIL import Image

# Directory containing images of known people
known_people_train_dir = "train_dataset/"
known_people_dir = "images/Indian_actors_faces/"

# Initialize lists to store face encodings and names
known_face_encodings = []
known_face_names = []
#
# # Iterate over each subdirectory (person) in the main directory
# for person_name in os.listdir(known_people_dir):
#     person_dir = os.path.join(known_people_dir, person_name)
#     # Check if it's a directory
#     if os.path.isdir(person_dir):
#         # Iterate over each file in the person's directory
#         for filename in os.listdir(person_dir):
#             image_path = os.path.join(person_dir, filename)
#             # Check if the file is a valid image file
#             try:
#                 with Image.open(image_path) as img:
#                     img.verify()  # Attempt to open and verify the image file
#
#                     # Convert PIL image to RGB format
#                     img_rgb = img.convert("RGB")
#
#                     # Convert RGB image to NumPy array
#                     img_np = np.array(img_rgb)
#
#                     # Detect faces in the image
#                     face_locations = face_recognition.face_locations(img_np)
#                     if face_locations:
#                         # Extract the first face found in the image using the bounding box coordinates
#                         top, right, bottom, left = face_locations[0]
#                         face_img = img_rgb.crop((left, top, right, bottom))
#
#                         # Perform data augmentation on the extracted face image
#                         output_path = os.path.join("train_dataset/", os.path.basename(person_dir))
#                         apply_blur(face_img, output_folder=output_path)
#                         apply_resize(face_img, output_folder=output_path)
#                         augment_data(face_img, output_folder=output_path)
#
#             except (IOError, SyntaxError):
#                 # Ignore any files that are not valid image files
#                 continue


# # Iterate over each subdirectory (person) in the main directory
# for person_name in os.listdir(known_people_dir):
#     person_dir = os.path.join(known_people_dir, person_name)
#     # Check if it's a directory
#     if os.path.isdir(person_dir):
#         # Iterate over each file in the person's directory
#         for filename in os.listdir(person_dir):
#             image_path = os.path.join(person_dir, filename)
#             # Check if the file is a valid image file
#             try:
#                 # Load the image
#                 image = cv2.imread(image_path)
#                 if image is None:
#                     raise IOError(f"Unable to load image from {image_path}")
#
#                 # Convert the image to RGB (MTCNN expects RGB format)
#                 rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#                 # Detect faces in the image using MTCNN
#                 faces = mtcnn.detect_faces(rgb_image)
#                 if faces:
#                     # Extract the first face detected
#                     face = faces[0]
#                     # Get the bounding box coordinates of the face
#                     x, y, w, h = face['box']
#
#                     # Ensure that the bounding box coordinates are valid
#                     left = max(x, 0)
#                     top = max(y, 0)
#                     right = min(x + w, image.shape[1])
#                     bottom = min(y + h, image.shape[0])
#
#                     # Verify bounding box coordinates
#                     print(f"Bounding box coordinates: (left={left}, top={top}, right={right}, bottom={bottom})")
#
#                     # Check if the bounding box is valid
#                     if right > left and bottom > top:
#                         # Crop the face region from the original image
#                         face_image = image[top:bottom, left:right]
#
#                         # Define output path for face images
#                         output_face_path = os.path.join("train_dataset/", os.path.basename(person_dir))
#
#                         # Save the face image
#                         cv2.imwrite(output_face_path, face_image)
#
#                         # Apply data augmentation on the face image
#                         apply_blur(output_face_path, output_folder=output_face_path)
#                         apply_resize(output_face_path, output_folder=output_face_path)
#                         augment_data(output_face_path, output_folder=output_face_path,
#                                      face_coordinates=(left, top, right, bottom))
#
#             except (IOError, SyntaxError) as e:
#                 print(f"Error processing {image_path}: {e}")
#                 continue

from mtcnn.mtcnn import MTCNN
import random
# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Iterate over each subdirectory (person) in the main directory
for person_name in os.listdir(known_people_dir):
    person_dir = os.path.join(known_people_dir, person_name)
    # Check if it's a directory
    if os.path.isdir(person_dir):
        # Create output directory for each person
        output_person_dir = os.path.join("train_dataset", person_name)
        os.makedirs(output_person_dir, exist_ok=True)
        # Iterate over each file in the person's directory
        count_of_images = len(os.listdir(person_dir))
        img_indexes = generate_unique_random_numbers(count_of_images-1,5)
        for i,filename in enumerate(os.listdir(person_dir)):
            image_path = os.path.join(person_dir, filename)
            # Check if the file is a valid image file
            try:
                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    raise IOError(f"Unable to load image from {image_path}")

                # Convert the image to RGB (MTCNN expects RGB format)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect faces in the image using MTCNN
                faces = mtcnn.detect_faces(rgb_image)
                if faces:
                    # Extract the first face detected
                    face = faces[0]
                    # Get the bounding box coordinates of the face
                    x, y, w, h = face['box']

                    # Ensure that the bounding box coordinates are valid
                    left = max(x, 0)
                    top = max(y, 0)
                    right = min(x + w, image.shape[1])
                    bottom = min(y + h, image.shape[0])

                    # Verify bounding box coordinates
                    print(f"Bounding box coordinates: (left={left}, top={top}, right={right}, bottom={bottom})")



                    # Check if the bounding box is valid
                    if right > left and bottom > top:
                        # Crop the face region from the original image
                        # face_image = image[top:bottom, left:right]

                        # Define output path for face images within the person's directory
                        output_face_path = os.path.join(output_person_dir, f"{filename}.jpg")

                        # Save the face image
                        cv2.imwrite(output_face_path, image)
                        if i in img_indexes:
                            # Apply data augmentation on the face image
                            apply_blur(output_face_path, output_folder=output_person_dir)
                            apply_resize(output_face_path, output_folder=output_person_dir)
                        augment_data(output_face_path, output_folder=output_person_dir,
                                     face_coordinates=(left, top, right, bottom),prefix=filename)
                        # augment_data(output_face_path, output_folder=output_person_dir,
                        #            face_coordinates=(left, top, right, bottom), should_add_jitter=False, prefix=f"no_jitter_{filename}")


            except (IOError, SyntaxError) as e:
                print(f"Error processing {image_path}: {e}")
                continue

for person_name in os.listdir(known_people_train_dir):
    person_dir = os.path.join(known_people_train_dir, person_name)
    # Check if it's a directory
    if os.path.isdir(person_dir):
        # Iterate over each file in the person's directory
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            print(image_path)
            # Check if the file is a valid image file
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Attempt to open and verify the image file
                    # Load the image file
                    person_image = face_recognition.load_image_file(image_path)
                    # Encode the face in the image
                    face_encoding = face_recognition.face_encodings(person_image)
                    # print(face_encoding)
                    face_encoding = face_encoding[0]

                    # Append the face encoding and name to the respective lists
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)
            except (IOError, SyntaxError,IndexError):
                # Ignore any files that are not valid image files
                continue
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = known_face_names[best_match_index]
            #
            # face_names.append(name)

            confidence = 0

            # Calculate face distances for all known face encodings
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # Find the index with the smallest distance (i.e., the best match)
            best_match_index = np.argmin(face_distances)

            # If the smallest distance is below a certain threshold (e.g., 0.6), consider it a match
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]

                confidence = (1 - face_distances[best_match_index]) * 100

            face_names.append((name, confidence))

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text,percentage = name
        text=f"{text}: {round(percentage)}"
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


