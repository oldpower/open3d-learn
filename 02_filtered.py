import open3d as o3d
import numpy as np


def StatisticalOutlierRemoval(filtered_pcd):
    """
    统计滤波参数：
    - nb_neighbors: 每个点用于计算距离的邻域点数
    - std_ratio: 标准差倍数，决定离群值的阈值
    """
    cl, ind = filtered_pcd.remove_statistical_outlier(nb_neighbors=10,
                                                      std_ratio=1)
    # 只保留内点（inliers）
    filtered_pcd_clean = filtered_pcd.select_by_index(ind)
    return filtered_pcd_clean

def RadiusOutlierRemoval(filtered_pcd):
    """
    半径滤波参数：
    - nb_points: 在给定半径内必须存在的最小点数
    - radius: 邻域搜索半径
    """
    cl, ind = filtered_pcd.remove_radius_outlier(nb_points=10, radius=10)
    # 只保留满足条件的点
    filtered_pcd_clean = filtered_pcd.select_by_index(ind)
    return filtered_pcd_clean


if __name__ == "__main__":
    # 读取点云数据
    pcd = o3d.io.read_point_cloud("demo.ply")  # 替换为你的点云文件路径
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

    # 点云滤波
    print("过滤前：",len(filtered_pcd.points))
    # filtered_pcd_clean = StatisticalOutlierRemoval(filtered_pcd)
    filtered_pcd_clean = RadiusOutlierRemoval(filtered_pcd)
    print("过滤后：",len(filtered_pcd_clean.points))

    # 可视化前的配置
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # 设置背景颜色，参数为RGB三元组，范围0-1
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]  # 深灰色背景
    # 设置点的大小
    opt.point_size = 2.0  # 根据需要调整大小
    # 将点云添加到可视化器中
    # vis.add_geometry(filtered_pcd)
    # 将两个点云加入可视化器（可选不同颜色区分）
    filtered_pcd.paint_uniform_color([1, 0, 0])     # 原始点云红色
    filtered_pcd_clean.paint_uniform_color([0, 1, 0])  # 去噪后绿色

    # vis.add_geometry(filtered_pcd)
    vis.add_geometry(filtered_pcd_clean)
    vis.add_geometry(filtered_pcd)

    # # 获取视角控制对象，并交换XY轴(通过旋转)
    # view_control = vis.get_view_control()
    # view_control.rotate(0, -90)  # 这里的值可能需要根据实际情况调整
    # 运行可视化器
    vis.run()
    vis.destroy_window()