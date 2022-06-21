# stream GraphNav robot pose
import rbd_spot

def main():
    conn = rbd_spot.SpotSDKConn(sdk_name="GraphNavPoseStreamer")
    graphnav_client = rbd_spot.graphnav.create_client(conn)
