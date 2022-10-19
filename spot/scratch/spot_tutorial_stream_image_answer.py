# Imports
import os
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Establish connection
    # 1. create SDK
    bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk('MyStreamImageClient')

    # 2. create robot
    hostname = os.environ.get('SPOT_IP', None)
    robot = sdk.create_robot(hostname)

    # 3. authenticate
    username = "user"
    password = os.environ.get('SPOT_USER_PASSWORD', None)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()

    # Lease is unnecessary! (because we are not controlling the robot)

    # Create Client & Call Service
    image_client = robot.ensure_client(ImageClient.default_service_name)

    while True:
        image_request = image_pb2.ImageRequest(image_source_name="frontleft_fisheye_image", quality=75)
        for image_response in image_client.get_image([image_request]):
            print(image_response.source)
            num_bytes = 1  # Assume a default of 1 byte encodings.
            if image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
                extension = ".png"
            else:
                if image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                    num_bytes = 3
                elif image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                    num_bytes = 4
                elif image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    num_bytes = 1
                elif image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                    num_bytes = 2
                dtype = np.uint8
                extension = ".jpg"

            img = np.frombuffer(image_response.shot.image.data, dtype=dtype)
            if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
                try:
                    # Attempt to reshape array into a RGB rows X cols shape.
                    img = img.reshape((image_response.shot.image.rows,
                                       image_response.shot.image.cols, num_bytes))
                except ValueError:
                    # Unable to reshape the image data, trying a regular decode.
                    img = cv2.imdecode(img, -1)
            else:
                img = cv2.imdecode(img, -1)

            plt.imshow(img, cmap='gray')
            plt.show(block=False)
            plt.pause(1)


if __name__ == "__main__":
    main()
