import sys
import os
import cv2
import numpy as np

from dx_engine import InferenceEngine as IE
from dx_engine import InferenceOption as IO

from engine.paddleocr import PaddleOcr
from engine.draw_utils import draw_with_poly_enhanced

TEXT_INDEX = 0             # Index for recognized text in OCR result
SCORE_INDEX = 1            # Index for confidence score in OCR result
CONFIDENCE_THRESHOLD = 0.3 # Minimum confidence score to display


class OCR():
    def __init__(self, ocr_workers=None):
        super().__init__()

        # Store OCR workers
        self.ocr_workers = ocr_workers if ocr_workers else []
        self.current_worker_index = 0  # Index for round-robin worker selection

        # Store screen dimensions for reference
        self.max_width = 1920
        self.max_height = 1080
        self.last_result_image = None
        self.last_result_text = None

    def get_next_worker(self):
        """
        Get next worker using round-robin selection
        """
        if not self.ocr_workers:
            return None

        worker = self.ocr_workers[self.current_worker_index]
        self.current_worker_index = (self.current_worker_index + 1) % len(self.ocr_workers)
        return worker

    def ocr_run(self, frame):
        """
        Run OCR inference on the given image
        
        Args:
            image: Input image for OCR processing
            file_path: Path to the image file
            
        Returns:
            tuple: (boxes, rotated_crops, rec_results)
        """
        worker = self.get_next_worker()
        if worker is None:
            print("No OCR workers available")
            return [], [], []

        try:
            boxes, rotated_crops, rec_results = worker(frame)
            return boxes, rotated_crops, rec_results
        except Exception as e:
            #print(f"Error during OCR inference: {e}")
            return [], [], []

    def ocr2image(self, org_image, boxes: list, rotated_crops: list, rec_results: list):
        """ 
        Convert OCR results to annotated image
        
        Args:
            org_image: Original input image
            boxes: Detection bounding boxes
            rotated_crops: Rotated crop regions
            rec_results: Recognition results
            
        Returns:
            numpy.ndarray: Annotated image with OCR results
        """
        from PIL import Image 
        image = org_image[:, :, ::-1]
        ret_boxes = [line for line in boxes]
        txts = [line['text'] for line in rec_results]
        bbox_text_poly_shape_quadruplets = []

        for i in range(len(ret_boxes)):
            bbox_text_poly_shape_quadruplets.append(
                ([np.array(ret_boxes[i]).flatten()], txts[i] if i < len(txts) else "", image.shape, image.shape)
            )

        im_sample = draw_with_poly_enhanced(image, bbox_text_poly_shape_quadruplets)
        return np.array(im_sample)



if __name__ == "__main__":
    '''
    @brief:
    @param:
        det_model_path: str, the path of the detection model : det.dxnn
        cls_model_path: str, the path of the classification model : cls.dxnn
        rec_model_path: str, the path of the recognition model
    @return:
        None
    '''
    det_model_path = "models/det_fixed.dxnn"
    cls_model_path = "models/cls_fixed.dxnn"
    rec_model_dirname = "models"

    window_name = "OCR"

    det_model = IE(det_model_path, IO().set_use_ort(True))
    cls_model = IE(cls_model_path, IO().set_use_ort(True))

    def make_rec_engines(model_dirname):
        io = IO().set_use_ort(True)
        rec_model_map = {}

        for ratio in [3, 5, 10, 15, 25, 35]:
            model_path = f"{model_dirname}/rec_fixed_ratio_{ratio}.dxnn"
            rec_model_map[ratio] = IE(model_path, io)
        return rec_model_map

    rec_models:dict = make_rec_engines(rec_model_dirname)
    ocr_workers = [PaddleOcr(det_model, cls_model, rec_models) for _ in range(1)]

    ocr = OCR(ocr_workers=ocr_workers)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    desired_width = 1920
    desired_height = 1080
    desired_fps = 5
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        crop_width = 440
        cropped_frame = frame[:, crop_width:desired_width-crop_width]


        # Check if the frame was read successfully
        if not ret:
            print("Error: Failed to grab frame.")
            break

        boxes, crops, rec_results = ocr.ocr_run(cropped_frame)
        ocr_frame = ocr.ocr2image(cropped_frame, boxes, boxes, rec_results)

        # Display the frame in a window
        cv2.imshow('OCR', cv2.cvtColor(ocr_frame, cv2.COLOR_RGB2BGR))

        # Wait for a key press (e.g., 'q' to quit)
        # cv2.waitKey(1) waits for 1 millisecond
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
