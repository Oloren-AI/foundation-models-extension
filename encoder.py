# Code adapted from the segment_anything.py file in the "anylabeling" repository by vietanhdev
# Original source: https://github.com/vietanhdev/anylabeling/blob/master/anylabeling/services/auto_labeling/segment_anything.py


import cv2
import numpy as np
import onnxruntime
from segment_anything import sam_model_registry, SamPredictor



def pre_process(input_size, max_width, max_height, image):
    # Resize by max width and max height
    # In the original code, the image is resized to long side 1024
    # However, there is a positional deviation when the image does not
    # have the same aspect ratio as in the exported ONNX model (2250x1500)
    # => Resize by max width and max height
    h, w = image.shape[:2]
    if w > max_width:
        h = int(h * max_width / w)
        w = max_width
    if h > max_height:
        w = int(w * max_height / h)
        h = max_height
    image = cv2.resize(image, (w, h))
    

    # Pad to have size at least max_width x max_height
    h, w = image.shape[:2]
    padh = max_height - h
    padw = max_width - w
    image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode="constant")
    

    # Normalize
    pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, -1)
    pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, -1)
    x = (image - pixel_mean) / pixel_std

    # Padding to square
    h, w = x.shape[:2]
    padh = input_size - h
    padw = input_size - w
    x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode="constant")
    x = x.astype(np.float32)

    # Transpose
    x = x.transpose(2, 0, 1)[None, :, :, :]

    encoder_inputs = {
        "x": x,
    }
    return encoder_inputs

def run_encoder(encoder_session, encoder_inputs):
    output = encoder_session.run(None, encoder_inputs)
    image_embedding = output[0]
    return image_embedding

if __name__ == '__main__':

    encoder_model_abs_path = "/Users/seabass/repos/github.com/oloren/foundation-models-extension/segment_anything_vit_b_encoder_quant.onnx"
    input_size = 1024
    max_width = 1024
    max_height = 682

    # Load models
    providers = onnxruntime.get_available_providers()

    # Pop TensorRT Runtime due to crashing issues
    # TODO: Add back when TensorRT backend is stable
    providers = [p for p in providers if p != "TensorrtExecutionProvider"]

    encoder_session = onnxruntime.InferenceSession(
        encoder_model_abs_path, providers=providers
    )

    image = cv2.imread('/Users/seabass/repos/github.com/oloren/foundation-models-extension/segment-attempt.png')
    # New way
    embeddings1 = run_encoder(encoder_session=encoder_session, encoder_inputs=pre_process(input_size=input_size, max_width=max_width, max_height=max_height, image=image))
    # Old way
    #checkpoint = "sam_vit_b_01ec64.pth"
    #model_type = "vit_b"
    #sam = sam_model_registry[model_type](checkpoint=checkpoint)
    #sam.to(device='cpu')
    #predictor = SamPredictor(sam)
    #predictor.set_image(image)
    # Old embedding
    #print("Old: ")
    #print(predictor.get_image_embedding().cpu().numpy())
    # New embedding
    print("New: ")
    print(embeddings1)
