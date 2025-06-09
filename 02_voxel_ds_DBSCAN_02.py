import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def detect_pointcloud_unit(points):
    """
    根据点云范围判断单位（mm / m）
    如果最大坐标值 > 1000，则认为是 mm；否则认为是 m
    """
    max_coord = np.max(np.abs(points))
    if max_coord > 1000:
        print(f"检测到点云单位为 mm（最大坐标: {max_coord:.2f}）")
        return 'mm'
    else:
        print(f"检测到点云单位为 m（最大坐标: {max_coord:.2f}）")
        return 'm'


def convert_to_meters(pcd, unit='mm'):
    """
    将点云转换为米制单位
    """
    points = np.asarray(pcd.points)
    if unit == 'mm':
        points = points / 1000.0
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def remove_noise_auto(pcd, scene_type='lidar'):
    """
    根据场景类型自动设置参数并去除噪声，支持单位检测和颜色保留
    """

    # Step 0: 检测单位并统一为米
    points = np.asarray(pcd.points)
    unit = detect_pointcloud_unit(points)
    pcd = convert_to_meters(pcd, unit=unit)

    # Step 1: 自动调参策略
    params = {
        'ground': {
            'voxel_size': 0.05,
            'eps': 0.3,
            'min_points': 10,
            'min_cluster_size': 50
        },
        'building': {
            'voxel_size': 0.1,
            'eps': 0.5,
            'min_points': 20,
            'min_cluster_size': 100
        },
        'vegetation': {
            'voxel_size': 0.05,
            'eps': 0.2,
            'min_points': 5,
            'min_cluster_size': 20
        },
        'lidar': {
            'voxel_size': 0.1,
            'eps': 0.3,
            'min_points': 10,
            'min_cluster_size': 50
        }
    }

    if scene_type not in params:
        print(f"未知场景类型 {scene_type}，使用默认参数：lidar")
        scene_type = 'lidar'

    cfg = params[scene_type]
    print(f"使用场景配置: {scene_type}, 参数: {cfg}")

    # Step 2: 下采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=cfg['voxel_size'])

    # Step 3: DBSCAN 聚类
    labels = np.array(
        downsampled_pcd.cluster_dbscan(eps=cfg['eps'], min_points=cfg['min_points'], print_progress=False)
    )

    # Step 4: 过滤小簇
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique_labels[counts >= cfg['min_cluster_size']]
    mask = np.isin(labels, valid_clusters)

    # Step 5: 提取干净点云
    cleaned_points = np.asarray(downsampled_pcd.points)[mask]

    # 保留颜色（如果有的话）
    if downsampled_pcd.has_colors():
        cleaned_colors = np.asarray(downsampled_pcd.colors)[mask]
    else:
        cleaned_colors = None

    cleaned_pcd = o3d.geometry.PointCloud()
    cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
    if cleaned_colors is not None:
        cleaned_pcd.colors = o3d.utility.Vector3dVector(cleaned_colors)

    return cleaned_pcd


def visualize_comparison(original_pcd, cleaned_pcd, window_width=800, window_height=600):
    """
    使用 Visualizer 类对比显示点云（可扩展性更强）
    """
    def show_pcd(pcd, title):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=window_width, height=window_height)
        opt = vis.get_render_option()
        # opt.background_color = [0.1, 0.1, 0.1]  # 深灰色背景
        opt.point_size = 2.0  # 设置点大小
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    print("显示原始点云...")
    show_pcd(original_pcd, "Original Point Cloud")

    print("显示去噪后点云...")
    show_pcd(cleaned_pcd, "Cleaned Point Cloud")


# 示例使用
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

    pcd = filtered_pcd
    # 打印基本信息
    print(f"原始点数量: {len(pcd.points)}")
    print(f"是否有颜色信息: {pcd.has_colors()}")

    # 使用自动去噪函数
    cleaned_pcd = remove_noise_auto(pcd, scene_type='vegetation')
    print(f"过滤后点数量: {len(cleaned_pcd.points)}")
    # 显示前后对比
    visualize_comparison(pcd, cleaned_pcd,window_width=800, window_height=500)

    # 保存结果（可选）
    output_file = "cleaned_pointcloud.ply"
    # o3d.io.write_point_cloud(output_file, cleaned_pcd)
    # print(f"去噪后点云已保存至: {output_file}")