# run spot SDK image streaming and record frequency
# for a few different combinations of image sources.
import os
import time
import numpy as np
import rbd_spot
import pandas as pd
from datetime import datetime

QUALITY=75
FORMAT=None

class TestCase:
    def __init__(self, name, sources):
        self.name = name
        self.sources = sources

    def run(self, conn, image_client,
            **kwargs):
        print(f"Running TestCase {self.name}")
        image_requests = rbd_spot.image.build_image_requests(
            self.sources,
            quality=kwargs.get("quality", 75),
            fmt=kwargs.get("format", None))

        duration = kwargs.get("duration", 10)
        _start_time = time.time()
        _all_times = []
        for result, time_taken in rbd_spot.image.getImageStream(image_client, image_requests):
            print(time_taken)
            _all_times.append(time_taken)
            _used_time = time.time() - _start_time
            if _used_time > duration:
                break
        return _all_times

def test_cases():
    all_sides_depth_visual = TestCase(
        "AllSidesDepthVisual",
        ["frontleft_depth_in_visual_frame",
         "frontright_depth_in_visual_frame",
         "left_depth_in_visual_frame",
         "right_depth_in_visual_frame",
         "back_depth_in_visual_frame"])

    two_depth_visual = TestCase(
        "TwoDepthVisual",
        ["frontleft_depth_in_visual_frame",
         "left_depth_in_visual_frame"])

    single_depth_visual = TestCase(
        "SingleDepthVisual",
        ["frontleft_depth_in_visual_frame"])

    all_sides_depth = TestCase(
        "AllSidesDepth",
        ["frontleft_depth",
         "frontright_depth",
         "left_depth",
         "right_depth",
         "back_depth"]
    )

    two_sides_depth = TestCase(
        "AllSidesDepth",
        ["frontleft_depth",
         "left_depth"]
    )

    single_depth = TestCase(
        "SingleDepth",
        ["frontleft_depth"]
    )

    return [all_sides_depth_visual,
            two_depth_visual,
            single_depth_visual,
            all_sides_depth,
            two_sides_depth,
            single_depth]


def main():
    print("Connecting to spot...")
    conn = rbd_spot.SpotSDKConn(sdk_name="StreamImageClient")
    print(f"Connected! ip: {conn.hostname}; type: {conn.conn_type}")
    image_client = rbd_spot.image.create_client(conn)

    rows = []
    for test_case in test_cases():
        all_times = test_case.run(conn, image_client,
                                  quality=QUALITY,
                                  fmt=FORMAT,
                                  duration=1)
        for time in all_times:
            rows.append([conn.conn_type, QUALITY, FORMAT, test_case.name, time])

    df = pd.DataFrame(rows,
                      columns=["conn_type", "quality", "format", "test_case", "response_time"])
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S)")
    result_file_name = f"streamingtimes_{conn.conn_type}_q{QUALITY}_f{FORMAT}_{timestamp}.csv"
    df.to_csv(os.path.join("results", result_file_name))

if __name__ == "__main__":
    main()
