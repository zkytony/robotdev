import numpy as np
import open3d as o3d
import rbd_spot

def extrinsic(camera_pose):
    """Given a camera_pose, returns a 4x4 extrinsic matrix that can transform
    a point in the world frame to the camera frame.

    The pose = (position, rotation), where position = (x,y,z)
    and rotation should be of length 3. It is specified
    by euler angles (degrees), same format as ai2thor rotation in pose.

    The job of the extrinsic matrix is to re-express the coordinates
    of a point in the world frame with respect to the camera frame.
    That means, the origin of the camera frame, which is at camera_pose,
    will become the new (0,0,0). The orientation of the camera frame,
    after recentering, should match its rotation in the world frame.

    This means the extrinsic matrix first applies translation to undo
    the position in the camera_pose, then applies the rotation specified
    in the camera pose.

    /author: Kaiyu Zheng
    """
    pos, rot = camera_pose
    x, y, z = pos
    pitch, yaw, roll = rot   # ai2thor convention
    # Unity documentation: In Unity these rotations are performed around the Z
    # axis, the X axis, and the Y axis, **in that order**; To express this
    # order in scipy rotation, we have to do 'yxz' (the order of matrix multiplication)
    Rmat = R_euler(yaw, pitch, roll, order='yxz').as_matrix()
    ex = np.zeros((4, 4))
    ex[:3, :3] = Rmat
    # not sure why z shouldn't be inverted. Perhaps another Unity thing.
    ex[:3, 3] = np.dot(Rmat, np.array([-x, -y, z]))
    ex[3, 3] = 1
    return ex

def extrinsic_inv(camera_pose):
    """Here, we are asked to produce a inverse extrinsic that transforms points
    from the camera frame to the world frame.
    /author: Kaiyu Zheng
    """
    return np.linalg.inv(extrinsic(camera_pose))

def open3d_pointcloud_from_rgbd(color, depth,
                                intrinsic, camera_pose=None,
                                depth_scale=1000.0,
                                depth_trunc=7,
                                convert_rgb_to_intensity=False):
    """
    /author: Kaiyu Zheng
    color (np.array): rgb image
    depth (np.array): depth image
    intrinsic: a tuple width, length, fx, fy, cx, cy
    camera_pose: a tuple (position, rotation) of the camera in world frame;
        position and rotation are tuples too.
    depth_scale: depth will be scaled by 1.0 / depth_scale
    depth_trunc: points with depth greater than depth_trunc will be discarded
    """
    width, height, fx, fy, cx, cy = intrinsic
    depth_img = o3d.geometry.Image(depth.astype(np.uint16))
    color_img = o3d.geometry.Image(color.astype(np.uint8))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img,
        depth_img,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    if camera_pose is not None:
        pcd.transform(extrinsic_inv(camera_pose))
    return pcd

def open3d_pointcloud_to_pcd(pcd, binary=True):
    """
    Given a point cloud (in Open3D format), returns an array of bytes
    that encodes the point cloud in the PCD file format.

    Args:
        pcd (open3d.geometry.PointCloud): a point cloud object
    """
    tmp_pcd_path = "/tmp/pointcloud.pcd"
    o3d.io.write_point_cloud(tmp_pcd_path, pcd,
                             write_ascii=not binary,
                             print_progress=True)
    with open(tmp_pcd_path, mode="rb") as f:
        return f.read()
