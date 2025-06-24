import base64
from io import BytesIO

from components.data_operations.dataset_api import dataset_shape
import matplotlib.pyplot as plt
import numpy as np

def match_shape(data_point, dataset_name):
    img_shape = dataset_shape(dataset_name)
    data_point = data_point.reshape(img_shape)
    return data_point

def encode_img_as_str(data_point):
    """
        passing dataset_name overrides img_shape
    """
    plt.figure(figsize=(4, 4), dpi=300)
    plt.imshow(data_point, cmap='gray')
    plt.axis('off')

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

