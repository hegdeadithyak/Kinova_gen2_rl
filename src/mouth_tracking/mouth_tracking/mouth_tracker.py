import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import cv_bridge
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import math

class MouthTracker(Node):
    def __init__(self):
        super().__init__('mouth_tracker')
        
        self.declare_parameter('mouth_point_topic', '/mouth_3d_point')
        mouth_point_topic = self.get_parameter('mouth_point_topic').value
        
        # MediaPipe initialization with high confidence to prevent false detections
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            # Increasing confidence to avoid "nostrils-as-eyes" at close range
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.8
        )
        
        # RealSense initialization
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            
            self.get_logger().info("RealSense D435i initialized with Robust Tracking.")
        except Exception as e:
            self.get_logger().error(f"Could not initialize RealSense: {e}")
            raise e
            
        self.bridge = cv_bridge.CvBridge()
        self.point_pub = self.create_publisher(PointStamped, mouth_point_topic, 10)
        self.debug_pub = self.create_publisher(Image, '/mouth_tracker/debug_image', 10)
        
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        
    def is_face_proportion_valid(self, landmarks):
        """
        Validate that the detected face has a realistic vertical layout.
        Prevents cases where nostrils are confused for eyes.
        """
        # Landmark indices:
        # 1: Nose tip
        # 33: Left eye (inner corner)
        # 263: Right eye (inner corner)
        # 13: Upper lip center
        
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        upper_lip = landmarks[13]
        
        # 1. Vertical check: Eyes MUST be above the nose tip, and nose tip above the lip
        # In image coords, Y increases downwards.
        if left_eye.y > nose_tip.y or right_eye.y > nose_tip.y:
            return False, "Eyes below nose"
            
        if nose_tip.y > upper_lip.y:
            return False, "Nose below mouth"
            
        # 2. Proportion check: Distance between eyes vs Eye-to-Mouth distance
        eye_dist = math.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
        eye_mouth_dist = upper_lip.y - (left_eye.y + right_eye.y)/2
        
        if eye_mouth_dist < eye_dist * 0.3: # Face is too vertically squashed
            return False, "Face squashed (nostrils-as-eyes risk)"
            
        return True, "Valid"

    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return
                
            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()
            
            # Mediapipe processing
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # VALIDATE FACE PROPORTIONS
                    valid, reason = self.is_face_proportion_valid(face_landmarks.landmark)
                    
                    if not valid:
                        # Draw warning on image
                        cv2.putText(display_image, f"REJECTED: {reason}", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        # self.get_logger().warn(f"Tracking rejected: {reason}")
                        break

                    # 1. DRAW COOL MASK (If valid)
                    self.mp_drawing.draw_landmarks(
                        image=display_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    self.mp_drawing.draw_landmarks(
                        image=display_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

                    # 2. EXTRACT MOUTH POINT
                    pt13 = face_landmarks.landmark[13]
                    pt14 = face_landmarks.landmark[14]
                    
                    h, w, c = color_image.shape
                    u_center = int((pt13.x + pt14.x) / 2 * w)
                    v_center = int((pt13.y + pt14.y) / 2 * h)
                    u_center = max(0, min(w - 1, u_center))
                    v_center = max(0, min(h - 1, v_center))
                    
                    # RealSense Min Depth is ~0.105m. Closer than that, readings are unreliable.
                    distance = depth_frame.get_distance(u_center, v_center)
                    
                    if distance > 0:
                        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u_center, v_center], distance)
                        x, y, z = point_3d
                        
                        msg = PointStamped()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.header.frame_id = "camera_color_optical_frame"
                        msg.point.x = float(x)
                        msg.point.y = float(y)
                        msg.point.z = float(z)
                        self.point_pub.publish(msg)
                        
                        # HUD
                        color = (0, 255, 255) if z > 0.15 else (0, 165, 255) # Warning color if very close
                        cv2.drawMarker(display_image, (u_center, v_center), color, 
                                      cv2.MARKER_CROSS, 20, 2)
                        cv2.putText(display_image, f"DIST: {z:.2f}m", (u_center + 10, v_center - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    break
            
            # Publish styled debug image
            debug_msg = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = "camera_color_optical_frame"
            self.debug_pub.publish(debug_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error in process_frame: {e}")

    def __del__(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    node = MouthTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
