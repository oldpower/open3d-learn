import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def preprocess_point_cloud(pcd, voxel_size=5):
    """点云预处理：降采样、去除离群点（毫米单位）"""
    # 体素网格降采样
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 统计离群点去除
    pcd_down, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 计算法线
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    
    return pcd_down

def segment_stem(pcd, distance_threshold=20, ransac_n=3, num_iterations=1000):
    """使用RANSAC拟合直线分割茶芽根茎（毫米单位）"""
    # RANSAC直线拟合
    line_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)
    
    # 提取直线内点（根茎部分）
    stem = pcd.select_by_index(inliers)
    stem.paint_uniform_color([0, 1, 0])  # 绿色表示根茎
    
    # 提取非直线部分（芽头和叶片）
    non_stem = pcd.select_by_index(inliers, invert=True)
    
    return stem, non_stem, line_model

def cluster_leaf_bud(non_stem, eps=15, min_points=30):
    """使用DBSCAN聚类区分芽头和叶片（毫米单位）"""
    # DBSCAN聚类
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(non_stem.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=True))
    
    # 找出最大的两个聚类（通常为芽头和叶片）
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        print("未找到足够的聚类")
        return None, None
    
    # 计算每个聚类的点数
    label_counts = [(label, np.sum(labels == label)) for label in unique_labels if label != -1]
    label_counts.sort(key=lambda x: x[1], reverse=True)
    
    # 假设最大的聚类是叶片，第二大的是芽头
    leaf_label = label_counts[0][0]
    bud_label = label_counts[1][0] if len(label_counts) > 1 else None
    
    # 提取叶片和芽头点云
    leaf_indices = np.where(labels == leaf_label)[0]
    leaf = non_stem.select_by_index(leaf_indices)
    leaf.paint_uniform_color([1, 0, 0])  # 红色表示叶片
    
    if bud_label is not None:
        bud_indices = np.where(labels == bud_label)[0]
        bud = non_stem.select_by_index(bud_indices)
        bud.paint_uniform_color([0, 0, 1])  # 蓝色表示芽头
    else:
        bud = None
    
    return leaf, bud

def visualize_results(stem, leaf, bud):
    """可视化分割结果"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 添加各个部分
    if stem is not None:
        vis.add_geometry(stem)
    if leaf is not None:
        vis.add_geometry(leaf)
    if bud is not None:
        vis.add_geometry(bud)
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    
    vis.run()
    vis.destroy_window()

def analyze_tea_sprout(pcd_path):
    """茶芽点云分析主流程"""
    # 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"原始点云点数: {len(pcd.points)}")
    
    # 预处理
    pcd_down = preprocess_point_cloud(pcd)
    print(f"降采样后点云点数: {len(pcd_down.points)}")
    
    # 分割根茎
    stem, non_stem, line_model = segment_stem(pcd_down)
    print(f"根茎点数: {len(stem.points)}")
    print(f"非根茎点数: {len(non_stem.points)}")
    print(f"拟合直线模型: {line_model}")  # [a,b,c,d]表示ax+by+cz+d=0
    
    # 聚类叶片和芽头
    leaf, bud = cluster_leaf_bud(non_stem)
    
    if leaf is not None:
        print(f"叶片点数: {len(leaf.points)}")
    if bud is not None:
        print(f"芽头点数: {len(bud.points)}")
    
    # 可视化结果
    visualize_results(stem, leaf, bud)
    
    return stem, leaf, bud, line_model


def validate_shape(pcd):
    """验证点云的几何形态特征（毫米单位），兼容不同Open3D版本"""
    if pcd is None or len(pcd.points) < 20:  # 增加最小点数要求
        return {
            'aspect_ratio': 0,
            'sphericity': 0,
            'cylindricity': 0,
            'valid': False
        }
    
    # 尝试添加微小扰动以解决共面问题
    points = np.asarray(pcd.points)
    perturbed_points = points + np.random.normal(0, 0.1, size=points.shape)  # 添加0.1mm的随机噪声
    pcd_perturbed = o3d.geometry.PointCloud()
    pcd_perturbed.points = o3d.utility.Vector3dVector(perturbed_points)
    
    try:
        # 计算点云包围盒
        aabb = pcd_perturbed.get_axis_aligned_bounding_box()
        
        # 尝试计算OBB，失败时回退到AABB
        obb = pcd_perturbed.get_oriented_bounding_box()
    except RuntimeError as e:
        print(f"警告: 计算OBB时出错: {e}. 使用AABB代替.")
        obb = aabb
    
    # 获取包围盒尺寸（兼容新旧API）
    try:
        # 新版本API (0.15.0+)
        dimensions = obb.extent
    except AttributeError:
        # 旧版本API
        dimensions = obb.get_extent()
    
    # 计算长宽高比（细长比）
    aspect_ratio = max(dimensions) / min(dimensions) if min(dimensions) > 0 else 0
    
    # 计算球形度指标（体积比）
    pcd_volume = len(pcd.points) * 1**3  # 假设每个点代表1mm³
    aabb_volume = dimensions[0] * dimensions[1] * dimensions[2]
    sphericity = pcd_volume / aabb_volume if aabb_volume > 0 else 0
    
    # 计算圆柱度指标（适用于根茎）
    if dimensions[0] > 0 and dimensions[1] > 0:
        # 假设根茎是圆柱体，计算圆柱度
        base_area = np.pi * min(dimensions[0], dimensions[1])**2 / 4
        cylinder_volume = base_area * max(dimensions)
        cylindricity = pcd_volume / cylinder_volume if cylinder_volume > 0 else 0
    else:
        cylindricity = 0
    
    return {
        'aspect_ratio': aspect_ratio,
        'sphericity': sphericity,
        'cylindricity': cylindricity,
        'valid': True
    }

if __name__ == "__main__":
    # 使用示例
    pcd_path = "demo_part.pcd"  # 替换为实际点云文件路径
    stem, leaf, bud, line_model = analyze_tea_sprout(pcd_path)
    
    # 提取根茎底部点作为采摘点
    if stem is not None:
        stem_points = np.asarray(stem.points)
        # 假设Z轴向下，取Z值最大的点作为根茎底部
        bottom_point_idx = np.argmax(stem_points[:, 2])
        picking_point = stem_points[bottom_point_idx]
        print(f"根茎底部采摘点坐标(mm): {picking_point}")

        stem_features = validate_shape(stem)
        print(f"根茎形态特征: {stem_features}")