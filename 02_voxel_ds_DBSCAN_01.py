import open3d as o3d
import numpy as np

def remove_noise_with_clustering(pcd, voxel_size=0.05, eps=0.3, min_points=10, min_cluster_size=50):
    # Step 1: 下采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Step 2: DBSCAN 聚类
    labels = np.array(downsampled_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    # Step 3: 过滤小簇
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique_labels[counts >= min_cluster_size]
    mask = np.isin(labels, valid_clusters)

    # Step 4: 提取干净点云
    cleaned_points = np.asarray(downsampled_pcd.points)[mask]
    cleaned_colors = np.asarray(downsampled_pcd.colors)[mask] if downsampled_pcd.has_colors() else None

    cleaned_pcd = o3d.geometry.PointCloud()
    cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
    if cleaned_colors is not None:
        cleaned_pcd.colors = o3d.utility.Vector3dVector(cleaned_colors)

    return cleaned_pcd

if __name__ == "__main__":
    # 加载点云
    file_path = "demo.ply"
    pcd = o3d.io.read_point_cloud(file_path)
    # 将点云数据转化为numpy数组
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # 获取颜色数据
    # 筛选Z轴小于设定阈值的点及其对应的颜色
    mask = (points[:, 2] > 500) & (points[:, 2] < 3000)
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    # 交换X轴和Y轴的数据
    filtered_points[:, [0, 1]] = filtered_points[:, [1, 0]]
    # 创建新的点云对象并赋值筛选后的点和颜色
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    cleaned_pcd = remove_noise_with_clustering(filtered_pcd, voxel_size=0.05*1000, eps=0.3*1000, min_points=10, min_cluster_size=50)
    o3d.visualization.draw_geometries([filtered_pcd],width=800, height=500)
    o3d.visualization.draw_geometries([cleaned_pcd],width=800, height=500)
