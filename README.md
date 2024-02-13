# Face Detection Project

This project demonstrates real-time face detection using OpenCV and face recognition using the face_recognition library. It includes functionalities for detecting faces in images, applying data augmentation, and recognizing faces in live video streams.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PIL (Python Imaging Library)
- face_recognition
- mtcnn

Install the required libraries using `pip install -r requirements.txt`.

## Usage

1. Run `main.py` to start the face detection and recognition application.
2. Ensure your webcam is connected and properly configured.
3. The application will detect and recognize faces in the live video stream.
4. Press 'q' to quit the application.

## Structure

- `main.py`: Main script for face detection and recognition.
- `utils.py`: Utility functions for image processing and data augmentation.

## Customization

- Modify `main.py` to customize the face recognition logic, such as adjusting confidence thresholds or adding new known faces.
- Update `utils.py` to enhance or modify the image processing and data augmentation functionalities.

## Acknowledgments

- This project uses the mtcnn library for face detection and the face_recognition library for face recognition.

