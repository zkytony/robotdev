#!/usr/bin/env python
# clear graph

import rbd_spot

def main():
    conn = rbd_spot.SpotSDKConn(sdk_name="GraphNavClearGraphClient")
    graphnav_client = rbd_spot.graphnav.create_client(conn)
    _, _used_time = rbd_spot.graphnav.clearGraph(graphnav_client)
    print("ClearGraph took %.3f" % _used_time)

if __name__ == "__main__":
    main()
