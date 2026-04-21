import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import pyrealsense2 as rs
import numpy as np

class StaticPointCloud(Node):
    def __init__(self):
        super().__init__('static_pointcloud_node')
        
        # 1. RealSense Setup
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.pc = rs.pointcloud()
            
            self.get_logger().info("RealSense D435i initialized on static stand.")
        except Exception as e:
            self.get_logger().error(f"Could not initialize RealSense: {e}")
            return
        
        # 2. ROS Comms
        self.pc_pub = self.create_publisher(PointCloud2, '/camera/point_cloud', 10)
        self.timer = self.create_timer(0.1, self.publish_pc)
        
    def publish_pc(self):
        try:
            # 1. Capture and Align
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return
            
            # 2. Generate PointCloud
            points = self.pc.calculate(depth_frame)
            self.pc.map_to(color_frame)
            
            # 3. Extract XYZ as a 2D float array
            # Structured array to standard 2D array
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # 4. Extract Colors
            color_img = np.asanyarray(color_frame.get_data())
            # Texture coordinates are [u, v] pairs
            tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
            
            h, w = color_img.shape[:2]
            
            # Map normalized UV to pixel coordinates
            u = np.clip((tex_coords[:, 0] * w).astype(int), 0, w - 1)
            v = np.clip((tex_coords[:, 1] * h).astype(int), 0, h - 1)
            
            # Efficiently sample colors
            colors_bgr = color_img[v, u]
            
            # Convert BGR to packed RGB for PointCloud2
            # Use bit manipulation on uint32 array
            r = colors_bgr[:, 2].astype(np.uint32)
            g = colors_bgr[:, 1].astype(np.uint32)
            b = colors_bgr[:, 0].astype(np.uint32)
            rgb_packed = (r << 16) | (g << 8) | b
            
            # 5. Filter valid points (where Z > 0)
            mask = vtx[:, 2] > 0
            vtx_filtered = vtx[mask]
            rgb_filtered = rgb_packed[mask]
            
            # 6. Build the message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "camera_link"
            
            # Combine XYZ and RGB into one structured array for creation
            # float32 x 3 + uint32 x 1
            dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)]
            cloud_data = np.empty(len(vtx_filtered), dtype=dtype)
            cloud_data['x'] = vtx_filtered[:, 0]
            cloud_data['y'] = vtx_filtered[:, 1]
            cloud_data['z'] = vtx_filtered[:, 2]
            cloud_data['rgb'] = rgb_filtered
            
            # Fields for RGB point cloud
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
            ]
            
            pc_msg = pc2.create_cloud(header, fields, cloud_data)
            self.pc_pub.publish(pc_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing point cloud: {e}")

    def __del__(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    node = StaticPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
