import rbd_spot

import asyncio
from typing import Tuple
from PIL import Image
from viam.components.camera import Camera
from viam.components.types import CameraMimeType


class SpotCamera(Camera):
    """
    Represents a single camera set (one stereo camera plus a fisheye camera)
    """

    def __init__(self, side):
        """
        We will stream frames from the grayscale image (fisheye camera), and
        stream point cloud using the depth map from the stereo. As a user,
        you only need to supply which camera set (e.g. frontleft) you'd like.

        Arguments:
            side (str): Which camera set this corresponds to; Could be:
                'frontleft', 'frontright', 'back', 'left', right'
        """
        super().__init__(f"spot-camera-{side}")
        self._side = side

        # Establish connection to spot
        print(f"establishing connection to camera set {side}...")
        self._conn = rbd_spot.SpotSDKConn(sdk_name="StreamImageClient")
        self._image_client = rbd_spot.image.create_client(self._conn)
        sources = [f"{side}_fisheye_image",
                   f"{side}_depth_in_visual_frame"]
        self._image_requests = rbd_spot.image.build_image_requests(sources)

    async def get_frame(self) -> Image.Image:
        result, time_taken = rbd_spot.image.getImage(self._image_client,
                                                     self._image_requests)
        print(time_taken)
        # return Image.fromarray(image)

    async def get_point_cloud(self) -> Tuple[bytes, str]:
        pass
    #     point_cloud = spot_camera.read_point_cloud()
    #     return (point_cloud.bytes, CameraMimeType.PCD.value)

def _test():
    spot_camera = SpotCamera("frontleft")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(spot_camera.get_frame())
    loop.close()

if __name__ == "__main__":
    _test()
