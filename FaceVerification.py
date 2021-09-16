from sys import modules
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class FaceVerification(object):
    """
    face verification class
    """
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def capture_user(self):
        # capturing and saving image from webcam
        save_path = ''
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        i = 0
        while i < 1:
            return_value, image = camera.read()
            cv2.imwrite(save_path, image)
            i += 1
        del(camera)
        cv2.destroyAllWindows()

    def create_box(self, image):
        image = cv2.imread(image)
        # creating a box around all the faces in an image
        faces = mtcnn.detect_faces(image)
        for face in faces:
            bounding_box = face['box']
            cv2.rectangle(image, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[0])+int(bounding_box[2]), int(bounding_box[1])+int(bounding_box[3])), (0, 0, 255), 2)
        return image

    def extract_face(self, image, resize=(224,224)):
        # extracting faces from an image
        image = cv2.imread(image)
        faces = mtcnn.detect_faces(image)
        for face in faces:
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height
            face_boundary = image[y1:y2, x1:x2]
            face_image = cv2.resize(face_boundary, resize)
        return face_image

    def get_embeddings(self, faces):
        # extracing all the features/embeddings from a face
        face = np.asarray(faces, 'float32')
        face = preprocess_input(face, version=2)
        model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
        return model.predict(face)

    def get_similarity(self, faces):
        # comparing embeddings between faces using cosine method
        embeddings = self.get_embeddings(faces)
        score = cosine(embeddings[0], embeddings[1])
        if score <= 0.5:
            return "Face Matched", score
        return "Face not Matched", score

mtcnn = MTCNN()
knownimg = ''
fcd = FaceVerification(mtcnn)

fcd.capture_user()
captured_image = ''

# cv2.imshow('', fcd.create_box(knownimg))
# cv2.waitKey(0)

# face_img = fcd.extract_face(knownimg)
# cv2.imshow('', face_img)
# cv2.waitKey(0)

faces = [fcd.extract_face(image) for image in [knownimg, captured_image]]

print(fcd.get_similarity(faces))