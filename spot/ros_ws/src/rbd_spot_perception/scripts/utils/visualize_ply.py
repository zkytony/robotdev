#!/usr/bin/env python
#
# visualizes the .ply point cloud using Open3d
# Usage:
#    visualize_ply.py <path_to_ply_file>
import sys
import open3d as o3d

def main():
    if len(sys.argv) < 1:
        print("Usage: visualize_ply.py <path_to_ply_file>")

    ply_file = sys.argv[1]
    print(f"Loading {ply_file}")
    pcd = o3d.io.read_point_cloud("ply_file")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
