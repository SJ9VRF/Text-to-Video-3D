# Utilities for image transformations

import cv2
import numpy as np

class TransformUtils:
    @staticmethod
    def apply_zoom(image, factor):
        """
        Apply zoom to an image based on the factor.
        :param image: Input image as a numpy array.
        :param factor: Zoom factor. Greater than 1.0 zooms in, less than 1.0 zooms out.
        :return: Zoomed image as a numpy array.
        """
        height, width = image.shape[:2]
        new_height, new_width = int(height * factor), int(width * factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if factor < 1.0:
            pad_height = (height - new_height) // 2
            pad_width = (width - new_width) // 2
            padded_image = cv2.copyMakeBorder(resized_image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return padded_image
        elif factor > 1.0:
            crop_height = (new_height - height) // 2
            crop_width = (new_width - width) // 2
            cropped_image = resized_image[crop_height:crop_height + height, crop_width:crop_width + width]
            return cropped_image
        else:
            return image

    @staticmethod
    def apply_rotation(image, angle):
        """
        Rotate the image by the specified angle.
        :param image: Input image as a numpy array.
        :param angle: Angle in degrees to rotate clockwise.
        :return: Rotated image as a numpy array.
        """
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return rotated_image

    @staticmethod
    def apply_translation(image, tx, ty):
        """
        Translate the image by tx and ty pixels.
        :param image: Input image as a numpy array.
        :param tx: Number of pixels to shift right.
        :param ty: Number of pixels to shift down.
        :return: Translated image as a numpy array.
        """
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return translated_image
