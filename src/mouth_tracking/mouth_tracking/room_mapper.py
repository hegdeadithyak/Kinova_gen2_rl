import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import pyrealsense2 as rs
import numpy as np
import pinocchio as pin
import os
import struct

class RoomMapper(Node):
    def __init__(self):
        super().__init__('room_mapper')
        
        # 1. Initialize Pinocchio for Kinova FK
        # Assuming robot.urdf is in the current working directory or kinova_ws root
        urdf_path = os.path.join(os.getcwd(), 'robot.urdf')
        if not os.path.exists(urdf_path):
             # Try parent directory if not in current
             urdf_path = os.path.join(os.path.dirname(os.getcwd()), 'robot.urdf')
             
        if not os.path.exists(urdf_path):
            self.get_logger().error(f"URDF not found at {urdf_path}. Please ensure robot.urdf is in the workspace root.")
            return

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.joint_names = [f'j2n6s300_joint_{i+1}' for i in range(6)]
        
        # 2. RealSense Setup
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.pc = rs.pointcloud()
            self.get_logger().info("RealSense D435i initialized for Room Mapping.")
        except Exception as e:
            self.get_logger().error(f"Could not initialize RealSense: {e}")
            return
        
        # 3. Map Storage (Voxel Grid for efficiency)
        self.world_points = {} # Key: (vx, vy, vz) grid coords, Value: [x, y, z, r, g, b]
        self.voxel_size = 0.05 # 5cm resolution for faster room mapping
        
        # 4. ROS Comms
        self.js_sub = self.create_subscription(JointState, '/joint_states', self.js_cb, 10)
        self.map_pub = self.create_publisher(PointCloud2, '/room_map', 10)
        
        # Timer for processing loop (5Hz is plenty for mapping)
        self.timer = self.create_timer(0.2, self.process_and_publish)
        
        self.q = pin.neutral(self.model)
        self.get_logger().info("Room Mapper Node initialized. Move the arm to scan!")

    def js_cb(self, msg):
        # Update joint positions for FK
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                # Find the index in the pinocchio model
                try:
                    jid = self.model.getJointId(name)
                    # For continuous joints (like Kinova), Pinocchio uses 2 slots (cos/sin) if not limited
                    # But the URDF usually defines them as continuous or revolute.
                    # We'll use the simple index mapping for now.
                    idx = self.model.joints[jid].idx_q
                    self.q[idx] = msg.position[i]
                except Exception:
                    pass

    def process_and_publish(self):
        try:
            # 1. Get Camera Pose in World via FK
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            
            # The camera is mounted on the wrist. 
            # In your URDF, 'j2n6s300_end_effector' is the tip.
            ee_id = self.model.getFrameId("j2n6s300_end_effector")
            if ee_id >= self.model.nframes:
                self.get_logger().error("End effector frame not found in URDF.")
                return
                
            oMee = self.data.oMf[ee_id]
            R = oMee.rotation
            t = oMee.translation
            
            # 2. Capture RealSense Points
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return
            
            # Generate point cloud
            points = self.pc.calculate(depth_frame)
            self.pc.map_to(color_frame)
            
            # Get vertices and texture coordinates (colors)
            vtx = np.asanyarray(points.get_vertices())
            # Convert to Nx3 numpy array
            pts_local = np.zeros((len(vtx), 3))
            pts_local[:, 0] = [v[0] for v in vtx]
            pts_local[:, 1] = [v[1] for v in vtx]
            pts_local[:, 2] = [v[2] for v in vtx]
            
            # 3. Transform Points to World Frame
            # Points are in camera frame (Z forward). 
            # Note: Depending on how the camera is mounted, you might need an extra 
            # static transform here between the 'end_effector' and the actual 'camera_link'.
            pts_world = (R @ pts_local.T).T + t
            
            # 4. Voxel Grid Filtering (Keep the map size manageable)
            # Decimate input: don't process every single pixel to save CPU
            step = 4 
            for i in range(0, len(pts_world), step):
                p = pts_world[i]
                
                # Filter out NaNs and distant points (> 5m)
                if np.isnan(p).any() or np.linalg.norm(pts_local[i]) > 5.0 or np.linalg.norm(pts_local[i]) < 0.1:
                    continue
                
                # Calculate voxel index
                v_idx = tuple((p / self.voxel_size).astype(int))
                
                if v_idx not in self.world_points:
                    self.world_points[v_idx] = p
            
            # 5. Publish the persistent Map
            if self.world_points:
                point_list = list(self.world_points.values())
                
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "world"
                
                msg = pc2.create_cloud_xyz32(header, point_list)
                self.map_pub.publish(msg)
                
                # self.get_logger().info(f"Map size: {len(self.world_points)} points", throttle_duration_sec=2.0)

        except Exception as e:
            self.get_logger().error(f"Error in Room Mapper loop: {e}")

    def __del__(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    node = RoomMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
