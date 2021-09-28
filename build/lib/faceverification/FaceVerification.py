from sys import modules
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

mtcnn = MTCNN()

def detect_face(image):
    """
    Given an image, checks if any face is present in the image
    :param image: image with faces
    :return: True/False
    """
    image = cv2.imread(image)
    faces = mtcnn.detect_faces(image)
    if faces:
        return True
    return False

def create_box(image):
    """
    Given an image, returns an image with a box around each face in the image
    :param image: image with faces
    :return: An image with a box around each face in the image
    """
    image = cv2.imread(image)
    # creating a box around all the faces in an image
    faces = mtcnn.detect_faces(image)
    for face in faces:
        bounding_box = face['box']
        cv2.rectangle(image, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[0])+int(bounding_box[2]), int(bounding_box[1])+int(bounding_box[3])), (0, 0, 255), 2)
    return image

def extract_face(image, resize=(224,224)):
    """
    Given an image, returns a resized image for the face in the image
    :param image: image with the face
    :return: A resized image with facial part and landmarks
    """
    image = cv2.imread(image)
    faces = mtcnn.detect_faces(image)
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_boundary = image[y1:y2, x1:x2]
        face_image = cv2.resize(face_boundary, resize)
        return face_image

def get_embeddings(faces):
    """
    Given an image, returns the 128-dimensional face encoding for each face in the image.
    :param faces: the images that contains one or more faces
    :return: A list of lists of face encodings (one for each face in the image)
    """
    face = np.asarray(faces, 'float32')
    face = preprocess_input(face, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
    return model.predict(face)

def compare_faces(faces):
    """
    Compare face encodings to see if they match.
    :param faces: the images that contains one or more faces
    :return: True/False
    """
    embeddings = get_embeddings(faces)
    score = cosine(embeddings[0], embeddings[1])
    if score <= 0.5:
        return True, score
    return False