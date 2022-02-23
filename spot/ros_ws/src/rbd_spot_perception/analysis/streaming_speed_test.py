# run spot SDK image streaming and record frequency
# for a few different combinations of image sources.
import os
import sys
import time
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import rbd_spot

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
        "TwoSidesDepth",
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


def run_test():
    print("Connecting to spot...")
    conn = rbd_spot.SpotSDKConn(sdk_name="StreamImageClient")
    print(f"Connected! ip: {conn.hostname}; type: {conn.conn_type}")
    image_client = rbd_spot.image.create_client(conn)

    rows = []
    for test_case in test_cases():
        all_times = test_case.run(conn, image_client,
                                  quality=QUALITY,
                                  fmt=FORMAT,
                                  duration=30)
        for time in all_times:
            rows.append([conn.conn_type, QUALITY, FORMAT, test_case.name, time])

    df = pd.DataFrame(rows,
                      columns=["conn_type", "quality", "format", "test_case", "response_time"])
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_file_name = f"streamingtimes_{conn.conn_type}_q{QUALITY}_f{FORMAT}_{timestamp}.csv"
    df.to_csv(os.path.join("results", result_file_name))


def plot_results(files_to_load):
    """plot results;
    We actually make the plotting code adapt to the non-optimal
    data frame schema produced by run_test, to save time / avoid
    rerunning the test."""
    def _test_name(test_case):
        if test_case.startswith("All"):
            num = 5
        if test_case.startswith("Two"):
            num = 2
        if test_case.startswith("Single"):
            num = 1
        if "Visual" in test_case:
            typ = "depth_visual"
        else:
            typ = "depth"
        return f"{typ}({num})"

    allres = []
    for filepath in files_to_load:
        allres.append(pd.read_csv(filepath))
    df = pd.concat(allres, axis=0)

    # We will add a column for whether the image source is 'depth' or 'depth visual'
    test_case_names = [_test_name(row['test_case'])
                   for _, row in df.iterrows()]
    df['test_case'] = test_case_names
    g = sns.boxplot(data=df,
                    x="test_case",
                    y="response_time",
                    hue="conn_type")

    plt.yticks(np.arange(round(min(df['response_time'])), round(max(df['response_time'])+0.5), 0.5))
    plt.ylabel("response time per request (s)")
    plt.xlabel("image type (No. cameras)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    mode = "plot"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "test":
        run_test()
    elif mode == "plot":
        plot_results(["./results/streamingtimes_spot_wifi_q75_fNone_20220223145741.csv",
                      "./results/streamingtimes_ethernet_q75_fNone_20220223152706.csv",
                      "./results/streamingtimes_rlab_q75_fNone_20220223154635.csv"])
    else:
        print('unknown mode', mode)
