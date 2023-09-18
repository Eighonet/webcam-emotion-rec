import cv2
import torch
import numpy as np

from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

MAX_PERSON = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device=device)
classes = fer.idx_to_class

def validate(x, y) -> tuple:
    return max(0, x), max(0, y)

def detect_face(frame) -> list:
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    if probs.all():
        bounding_boxes = bounding_boxes[probs > 0.6]
        return bounding_boxes
    else:
        return []

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam.")

class_prev = [-1 for i in range(MAX_PERSON)]

while True:
    ret, frame_o = cap.read()
    frame = cv2.cvtColor(frame_o, cv2.COLOR_BGR2RGB)

    bounding_boxes = detect_face(frame)

    sorted(bounding_boxes, key=lambda bbox: bbox[0])

    for i, bbox in enumerate(bounding_boxes):
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        face_img = frame[y1:y2, x1:x2, :]

        try:
            _, scores = fer.predict_emotions(face_img, logits=False)
        except:
            continue

        cv2.putText(frame_o, f'Person {i}', validate(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (115, 255, 115), 2)
        for j, score in enumerate(scores):
            if np.argmax(scores) == j:
                color = (0, 0, 255)

                if classes[j] != class_prev[i]:
                    print(f'Person {i}', classes[np.argmax(scores)], max(scores))
                    class_prev[i] = classes[j]
            else:
                color = (115, 255, 115)
            cv2.putText(frame_o, f'{classes[j]}: {str(np.round(score, 2))}', validate(x2 + 10, y1 + 30 * j + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.rectangle(frame_o, (x1, y1), (x2, y2), (115, 255, 115), 2)
    cv2.imshow('Window 1', frame_o)

    c = cv2.waitKey(1)
    if c == 27:  # press ESC to close the window
        break

cap.release()
cv2.destroyAllWindows()
