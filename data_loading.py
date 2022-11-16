import os
import h5py
import numpy as np
import open3d as o3d


dataDir = "."
hdf5Files = ["training_pointcloud_hdf5", "testing_pointcloud_hdf5"]
dataTypes = ["synthetic", "real"]
partitions = ["Training", "Validation"]

classLabels = {0:"Normal", 1:"Displacement", 2:"Brick", 3:"Rubber Ring"}

for h5 in hdf5Files:
    for dt in dataTypes:
        path = os.path.join(dataDir, "{}_{}.h5".format(h5, dt))

        with h5py.File(path, 'r') as hdf:          
            if h5 == "training_pointcloud_hdf5":
                partitions = ["Training", "Validation"]
            else:
                partitions = ["Testing"]

            for partition in partitions:
                data = np.asarray(hdf[f'{partition}/PointClouds'][:])
                labels = np.asarray(hdf[f'{partition}/Labels'][:])

                uniqueLabels, uniqueCounts = np.unique(labels, return_counts = True)

                
                print(f'\nFilepath: {path}')
                print(f'[{partition} Data]')
                print(data.shape)
                print(len(data))
                print(data[0])
                xyz = data[0]
                print(xyz.shape)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                o3d.io.write_point_cloud("./testdata.ply",pcd)
                print(labels[2])
                # print(f'Data Shape: {data.shape} | Type: {data[0].dtype}')
                # print(f'Label Shape: {labels.shape} | Type: {labels[0].dtype}')
                # print(f'Labels: {uniqueLabels} | Counts: {uniqueCounts}\n')