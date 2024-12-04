import numpy as np
import cv2

def normalize_data(data, mean=None, std=None):
    if mean is None or std is None:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
    return (data - mean) / std, mean, std

def resize_images(images, size=(128, 128)):
    return np.array([cv2.resize(img, size) for img in images])

def apply_clahe(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return np.array([clahe.apply(img) for img in images])

def augment_images(images, rotation_range=15):
    augmented = []
    for img in images:
        angle = np.random.uniform(-rotation_range, rotation_range)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        augmented.append(cv2.warpAffine(img, M, (w, h)))
    return np.array(augmented)

def process_insar_images(images):
    resized = resize_images(images)
    normalized, mean, std = normalize_data(resized)
    return apply_clahe(normalized)
