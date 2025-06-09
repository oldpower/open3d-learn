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

def ProjectToPlane(pcd, plane_model):
    """
    将地面点投影到拟合出的平面上
    """
    points = np.asarray(pcd.points)
    a, b, c, d = plane_model

    # 计算点到平面的距离
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # 点投影到平面公式：P_proj = P - ((n·P + d)/||n||^2) * n
    t = (np.dot(points, normal) + d) / (np.linalg.norm(normal) ** 2)
    proj_points = points - t[:, None] * normal

    # 创建新的点云对象
    pcd_projected = pcd.select_by_index([])
    pcd_projected.points = o3d.utility.Vector3dVector(proj_points)

    # 保留颜色（如果有的话）
    if pcd.has_colors():
        pcd_projected.colors = pcd.colors

    return pcd_projected

def reconstruct_outdoor_scene(pcd, 
                              radius_list=[0.05, 0.1, 0.2], 
                              alpha=0.1,
                              normal_radius=0.1,
                              max_nn=30,
                              simplify=True,
                              target_triangles=100000,
                              smooth=True,
                              number_of_iterations=5):
    """
    对室外复杂点云进行三维重建，适用于包含植物、树叶、地面等场景。
    
    参数:
        pcd: 输入点云 (open3d.geometry.PointCloud)
        radius_list: BPA使用的球半径列表
        alpha: Alpha Shape 的参数（可选）
        normal_radius: 法线估计搜索半径
        max_nn: 法线估计最大邻域数
        simplify: 是否简化网格
        target_triangles: 简化后的三角形数量
        smooth: 是否平滑网格
        number_of_iterations: 平滑迭代次数
        
    返回:
        mesh: 重建的三角网格模型
    """
    
    # Step 1: 预处理 - 估计法线（必须步骤）
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn)
    )
    pcd.orient_normals_towards_camera_location()  # 统一法线方向

    # Step 2: 使用 Ball Pivoting Algorithm (BPA) 进行重建
    print("Reconstructing with Ball Pivoting Algorithm...")
    radii = o3d.utility.DoubleVector(radius_list)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

    # Step 3: 可选 - 使用 Alpha Shape 作为备选方案（如果 BPA 效果不佳）
    if len(mesh.triangles) == 0:
        print("BPA failed to generate mesh, trying Alpha Shape...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)

    # Step 4: 网格简化（加速渲染）
    if simplify and len(mesh.triangles) > 0:
        print(f"Simplifying mesh to {target_triangles} triangles...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    # Step 5: 网格平滑（提高视觉效果）
    if smooth and len(mesh.triangles) > 0:
        print(f"Smoothing mesh for {number_of_iterations} iterations...")
        mesh.filter_smooth_simple(number_of_iterations=number_of_iterations)
        mesh.compute_vertex_normals()
    return mesh

if __name__ == "__main__":
    # 读取点云数据
    pcd = o3d.io.read_point_cloud("demo.ply")  # 替换为你的点云文件路径
    # 将点云数据转化为numpy数组
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # 获取颜色数据
    # 筛选Z轴小于设定阈值的点及其对应的颜色
    mask = points[:, 2] < 3000
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
    # filtered_pcd = StatisticalOutlierRemoval(filtered_pcd)
    filtered_pcd = RadiusOutlierRemoval(filtered_pcd)
    print("过滤后：",len(filtered_pcd.points))




    ############################################################################################
    # 平面拟合
    # 使用RANSAC算法进行平面分割
    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=10,
                                                            ransac_n=3,
                                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = filtered_pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 将地面点云涂成红色
    non_inlier_cloud = filtered_pcd.select_by_index(inliers, invert=True)

    # # 可视化结果
    # o3d.visualization.draw_geometries([inlier_cloud])

    # 执行投影
    smoothed_ground = ProjectToPlane(inlier_cloud, plane_model)

    # Step 3: 合并地面和平滑后的点云
    final_pcd = non_inlier_cloud + smoothed_ground
    print("合并后：",len(final_pcd.points))


    ############################################################################################
    # 估计法线（必须步骤）
    pcd = filtered_pcd
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location()  # 统一法线方向

    # 泊松重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # # ✅ 正确方式：基于每个三角形的顶点密度计算平均密度
    # vertex_densities = np.asarray(densities)  # 每个顶点的密度
    # triangles = np.asarray(mesh.triangles)   # 每个三角形由三个顶点索引组成
    # # 对每个三角形，取其三个顶点的平均密度作为该三角形的密度
    # triangle_density = vertex_densities[triangles].mean(axis=1)
    # # 设置阈值（如前25%）
    # density_threshold = np.quantile(triangle_density, 0.25)
    # print(f"Thresholding using density threshold: {density_threshold}")
    # # 构建 mask：保留高于阈值的三角形
    # triangles_to_keep = triangle_density > density_threshold
    # # ✅ 应用 mask 到网格
    # mesh.remove_triangles_by_mask(triangles_to_keep)
    # # 可选：清理无效顶点和边
    # mesh.remove_unreferenced_vertices()
    # mesh.remove_degenerate_triangles()

    # 简化网格（可选）
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)

    # # 执行重建
    # mesh = reconstruct_outdoor_scene(filtered_pcd,
    #                                 radius_list=[0.05, 0.1, 0.2],
    #                                 normal_radius=0.1,
    #                                 max_nn=30,
    #                                 simplify=True,
    #                                 smooth=True)


    ############################################################################################
    # 可视化前的配置
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # 设置背景颜色，参数为RGB三元组，范围0-1
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]  # 深灰色背景
    # 设置点的大小
    opt.point_size = 2.0  # 根据需要调整大小
    # 将点云添加到可视化器中
    vis.add_geometry(filtered_pcd)
    # # 获取视角控制对象，并交换XY轴(通过旋转)
    # view_control = vis.get_view_control()
    # view_control.rotate(0, -90)  # 这里的值可能需要根据实际情况调整
    # 运行可视化器
    vis.run()
    vis.destroy_window()