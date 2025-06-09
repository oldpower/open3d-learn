## 1、open3d简介
`open3d`是一个开源的、高效的3D数据处理库，支持点云、网格等多种几何类型。
- **功能**：
  - 点云读写（ply, pcd等格式）
  - 滤波、降采样、配准、配色
  - 可视化
  - 几何变换、法线估计、平面分割等
- **官网**：https://www.open3d.org/
- **安装**：
  ```bash
  pip install open3d
  ```

## 2、可视化代码
### 2.1 基础可视化
    完成点云文件读取和可视化
```python
import open3d as o3d
# 读取点云
pcd = o3d.io.read_point_cloud("demo.ply")
# 可视化
o3d.visualization.draw_geometries([pcd])
```
![base](./image_best.png#pic_center)

### 2.2 进阶可视化
    a. 一些点云文件呈现锥形，这通常是通过一些设备获取的点云数据，较远的点基本属于无效点，影响可视化，因此可以根据深度z选取某一范围的点。
    b. 有些设备记录的点云，x和y是镜像相反的，需要调整。
    c. 设置可视化背景、点的大小、初始角度参数。
```python
import open3d as o3d
if __name__ == "__main__":
    # 读取点云数据
    pcd = o3d.io.read_point_cloud("demo.ply") 

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
    # # 获取视角控制对象，调整角度
    # view_control = vis.get_view_control()
    # view_control.rotate(0, -90)  # 这里的值可能需要根据实际情况调整
    # 运行可视化器
    vis.run()
    vis.destroy_window()
```
**和基础可视化结果对比，下面是恢复点云xy镜像，设定深度z范围，将背景设置为深灰色，点大小为2的可视化效果：**
![base](./image_advance.png#pic_center)


