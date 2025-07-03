import cv2
from mtcnn import MTCNN

# Load the MTCNN detector
detector = MTCNN()

# Load the image
image = cv2.imread('Test image.jpg')
if image is None:
    print("Error: Could not read the image.")
    exit()

# Detect faces using MTCNN
results = detector.detect_faces(image)

# Check if any faces were detected
if len(results) == 0:
    print("No face detected in the image.")
else:
    # Loop through detected faces
    for result in results:
        x, y, width, height = result['box']
        confidence = result['confidence']

        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Add confidence score
        cv2.putText(
            image,
            f"{confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save the output image
    cv2.imwrite('output_cnn.jpg', image)

    # Display the result
    cv2.imshow('Detected Face', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
