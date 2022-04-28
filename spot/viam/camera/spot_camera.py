import asyncio
from typing import Tuple
import numpy as np
from PIL import Image

# path to rbd_spot (needed for working with spot)
import os
import sys
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ABS_PATH, '../../ros_ws/src/rbd_spot/src'))
sys.path.insert(0, os.path.join(ABS_PATH, '../../ros_ws/src/rbd_spot_robot/src'))
sys.path.insert(0, os.path.join(ABS_PATH, '../../ros_ws/src/rbd_spot_perception/src'))

import rbd_spot
from viam.components.camera import Camera
from viam.components.types import CameraMimeType

from . import utils

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
        fisheye_response = result[0]
        return Image.fromarray(rbd_spot.image\
                               .image_response_to_array(self._conn, fisheye_response))

    async def get_point_cloud(self) -> Tuple[bytes, str]:
        result, time_taken = rbd_spot.image.getImage(self._image_client,
                                                     self._image_requests)
        print(time_taken)
        fisheye_response = result[0]
        fisheye_img = rbd_spot.image.image_response_to_array(
            self._conn, fisheye_response)
        # extend single channel to three channels (otherwise gets unsupported
        # format for open3d)
        fisheye_img = np.stack((fisheye_img,)*3, axis=2).astype(np.uint8)
        depth_visual_response = result[1]
        depth_visual_img = rbd_spot.image.image_response_to_array(
            self._conn, depth_visual_response)
        intrinsic = rbd_spot.image.extract_pinhole_intrinsic(depth_visual_response)
        point_cloud = utils.open3d_pointcloud_from_rgbd(fisheye_img, depth_visual_img, intrinsic)
        # For now, let's use open3d's io function to convert the point cloud
        # into pcd; because the PCD file format is sophisticated:
        # https://github.com/isl-org/Open3D/blob/master/cpp/open3d/io/file_format/FilePCD.cpp#L791
        pcd_binary = utils.open3d_pointcloud_to_pcd(point_cloud)
        return (pcd_binary, CameraMimeType.PCD.value)


def _test_local():
    spot_camera = SpotCamera("frontleft")

    print("testing get_frame...")
    loop = asyncio.new_event_loop()
    img = loop.run_until_complete(spot_camera.get_frame())
    img.save("test.png")


    print("testing get_point_cloud...")
    pcd_binary, mimetype = loop.run_until_complete(spot_camera.get_point_cloud())
    loop.close()

    with open("/tmp/pointcloud_frame.pcd", "wb") as f:
        f.write(pcd_binary)
    import open3d as o3d
    pcd = o3d.io.read_point_cloud("/tmp/pointcloud_frame.pcd")
    viz = o3d.visualization.Visualizer()
    viz.create_window()
    viz.add_geometry(pcd)
    opt = viz.get_render_option()
    opt.show_coordinate_frame = True
    viz.run()
    viz.destroy_window()

if __name__ == "__main__":
    _test_local()
