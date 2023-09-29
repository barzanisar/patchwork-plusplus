import os
import sys
import open3d as o3d
import numpy as np
# import pypatchworkpp

cur_dir = os.path.dirname(os.path.abspath(__file__))
input_cloud_filepath = os.path.join(cur_dir, '../../data/waymo/0000.npy')

try:
    patchwork_module_path = os.path.join(cur_dir, "../../build/python_wrapper")
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)

def read_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    return scan

def read_npy(npy_path):
    scan = np.load(npy_path)
    scan[:,3] = np.tanh(scan[:, 3]) * 255.0

    return scan[:,:4]

if __name__ == "__main__":

    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
    params.verbose = True
    params.enable_RNR = True
    params.enable_RVPF = True
    params.enable_TGR = True #impo

    params.num_iter = 3              # Number of iterations for ground plane estimation using PCA.
    params.num_lpr = 50              # Maximum number of points to be selected as lowest points representative.
    params.num_min_pts = 10          # Minimum number of points to be estimated as ground plane in each patch.
    params.num_zones = 4             # Setting of Concentric Zone Model(CZM)
    params.num_rings_of_interest = 4 # Number of rings to be checked with elevation and flatness values.

    params.RNR_ver_angle_thr = -15.0 # Noise points vertical angle threshold. Downward rays of LiDAR are more likely to generate severe noise points.
    params.RNR_intensity_thr = 0.2   # Noise points intensity threshold. The reflected points have relatively small intensity than others.

    params.sensor_height = 0                 
    params.th_seeds = 0.5                        # threshold for lowest point representatives using in initial seeds selection of ground points.
    params.th_dist = 0.125                       # threshold for thickenss of ground.
    params.th_seeds_v = 0.25                     # threshold for lowest point representatives using in initial seeds selection of vertical structural points.
    params.th_dist_v = 0.1                       # threshold for thickenss of vertical structure.
    params.max_range = 80.0                      # max_range of ground estimation area
    params.min_range = 0.5                       # min_range of ground estimation area
    params.uprightness_thr = 0.707               # threshold of uprightness using in Ground Likelihood Estimation(GLE). Please refer paper for more information about GLE.
    params.adaptive_seed_selection_margin = -1.2 # parameter using in initial seeds selection

    params.num_sectors_each_zone = [16, 32, 54, 32] # Setting of Concentric Zone Model(CZM)
    params.num_rings_each_zone = [2, 4, 4, 4]       # Setting of Concentric Zone Model(CZM)

    params.max_flatness_storage = 1000  # The maximum number of flatness storage
    params.max_elevation_storage = 1000 # The maximum number of elevation storage
    params.elevation_thr = [0, 0, 0, 0] # threshold of elevation for each ring using in GLE. Those values are updated adaptively.
    params.flatness_thr = [0, 0, 0, 0]  # threshold of flatness for each ring using in GLE. Those values are updated adaptively.

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    # Load point cloud
    pointcloud = read_npy(input_cloud_filepath)

    # Estimate Ground
    PatchworkPLUSPLUS.estimateGround(pointcloud)

    # Get Ground and Nonground
    ground      = PatchworkPLUSPLUS.getGround()
    nonground   = PatchworkPLUSPLUS.getNonground()
    time_taken  = PatchworkPLUSPLUS.getTimeTaken()

    # Get centers and normals for patches
    centers     = PatchworkPLUSPLUS.getCenters()
    normals     = PatchworkPLUSPLUS.getNormals()

    print("Origianl Points  #: ", pointcloud.shape[0])
    print("Ground Points    #: ", ground.shape[0])
    print("Nonground Points #: ", nonground.shape[0])
    print("Time Taken : ", time_taken / 1000000, "(sec)")
    print("Press ... \n")
    print("\t H  : help")
    print("\t N  : visualize the surface normals")
    print("\tESC : close the Open3D window")

    # Visualize
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width = 600, height = 400)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    ground_o3d = o3d.geometry.PointCloud()
    ground_o3d.points = o3d.utility.Vector3dVector(ground)
    ground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float) # RGB
    )

    nonground_o3d = o3d.geometry.PointCloud()
    nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
    nonground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float) #RGB
    )

    centers_o3d = o3d.geometry.PointCloud()
    centers_o3d.points = o3d.utility.Vector3dVector(centers)
    centers_o3d.normals = o3d.utility.Vector3dVector(normals)
    centers_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float) #RGB
    )

    vis.add_geometry(mesh)
    vis.add_geometry(ground_o3d)
    vis.add_geometry(nonground_o3d)
    vis.add_geometry(centers_o3d)

    vis.run()
    vis.destroy_window()
