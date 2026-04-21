#!/usr/bin/env python3 # Shebang line telling the OS to execute this script using the Python 3 interpreter

import math # Import the standard Python math library for mathematical operations like square root

import rclpy # Import the main ROS 2 Python client library
import rclpy.time # Import the ROS 2 time module for handling ROS time objects
import rclpy.duration # Import the ROS 2 duration module for handling time intervals
from builtin_interfaces.msg import Duration # Import the standard Duration message type for trajectory timing
from control_msgs.action import FollowJointTrajectory # Import the action type used to command the robot arm trajectory
from geometry_msgs.msg import Point, Pose, Quaternion, PointStamped # Import geometry message types for 3D points, orientations, and poses
from moveit_msgs.msg import RobotState # Import the RobotState message type to represent the robot's kinematic state
from moveit_msgs.srv import GetCartesianPath # Import the service type used to ask MoveIt to compute a linear path
from rclpy.action import ActionClient # Import ActionClient to send goals to ROS 2 action servers
from rclpy.callback_groups import ReentrantCallbackGroup # Import ReentrantCallbackGroup to allow parallel callback execution
from rclpy.executors import MultiThreadedExecutor # Import MultiThreadedExecutor to run the node with multiple threads
from rclpy.node import Node # Import the base Node class required to create a ROS 2 node
from sensor_msgs.msg import JointState # Import the JointState message type to read the current angles of the robot joints
from std_srvs.srv import Trigger # Import the Trigger service type for a simple "go" command with no arguments
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException # Import TF2 classes and exceptions for coordinate frame math
from tf2_geometry_msgs import do_transform_point # Import the utility function to apply TF2 spatial transformations to points
from visualization_msgs.msg import Marker # Import the Marker message type to draw shapes in the RViz 3D viewer


SPOON_LENGTH_M = 0.05 # Define the physical length of the spoon in meters (5 cm)
SAFETY_GAP_M   = 0.05 # Define an extra safety buffer distance from the mouth in meters (5 cm)
TOTAL_OFFSET_M = SPOON_LENGTH_M + SAFETY_GAP_M   # Calculate total standoff distance: 0.10 m (10 cm)

WORLD_FRAME = 'world' # Define the name of the static global reference frame
CAM_FRAME   = 'camera_color_optical_frame' # Define the name of the reference frame originating from the camera lens
EE_LINK     = 'j2s6s200_end_effector' # Define the name of the robot's end-effector (hand/tool) frame
ARM_GROUP   = 'arm' # Define the MoveIt planning group name for the manipulator

CART_STEP  = 0.01    # Define the maximum distance (1 cm) between points in the computed Cartesian path
FEED_SPEED = 0.15    # Define the execution speed for the arm movement in meters per second
MIN_DT     = 0.02    # Define the minimum time delta (seconds) between trajectory waypoints to respect acceleration limits
MAX_REACH  = 0.85    # Define the maximum safe reachable radius of the Kinova j2s6s200 arm in meters



class MouthFeedingPlanner(Node): # Define the main ROS 2 node class inheriting from rclpy.node.Node
    def __init__(self): # Define the constructor method for the node initialization
        super().__init__('mouth_feeding_planner') # Initialize the parent Node class with the name 'mouth_feeding_planner'
        self._cb = ReentrantCallbackGroup() # Create a reentrant callback group to allow overlapping callback executions

        self._tf_buf = Buffer() # Initialize a TF2 buffer to store coordinate frame transformations over time
        self._tf_listener = TransformListener(self._tf_buf, self) # Initialize a TF2 listener to populate the buffer with incoming transforms

        self._latest_mouth: PointStamped | None = None # Initialize a class variable to store the latest perceived mouth position
        self.create_subscription( # Create a ROS 2 subscriber to listen for mouth coordinates
            PointStamped, '/mouth_3d_point', # Specify the message type and the topic name to subscribe to
            self._mouth_cb, 10, callback_group=self._cb, # Assign the callback function, queue size of 10, and the callback group
        ) # Close the create_subscription call

        self._latest_js: JointState | None = None # Initialize a class variable to store the latest robot joint states
        self.create_subscription( # Create a ROS 2 subscriber to listen for robot joint states
            JointState, '/joint_states', # Specify the message type and the topic name to subscribe to
            self._js_cb, 10, callback_group=self._cb, # Assign the callback function, queue size of 10, and the callback group
        ) # Close the create_subscription call

        self._marker_pub = self.create_publisher(Marker, '/feeding_marker', 10) # Create a publisher to send visual markers to RViz

        self._cart_client = self.create_client( # Create a ROS 2 service client to request Cartesian path planning
            GetCartesianPath, '/compute_cartesian_path', # Specify the service type and the MoveIt service topic name
            callback_group=self._cb, # Assign the service client to the reentrant callback group
        ) # Close the create_client call
        self._traj_client = ActionClient( # Create an Action Client to send trajectory execution goals to the arm controller
            self, FollowJointTrajectory, # Pass the node instance and the action type
            '/arm_controller/follow_joint_trajectory', # Specify the action server topic name
            callback_group=self._cb, # Assign the action client to the reentrant callback group
        ) # Close the ActionClient initialization

        self._busy = False # Initialize a boolean flag to track if a feeding motion is currently in progress
        self.create_service( # Create a ROS 2 service server to trigger the feeding action
            Trigger, '/feed_trigger', # Specify the service type and the topic name for the trigger
            self._trigger_cb, callback_group=self._cb, # Assign the service callback function and the callback group
        ) # Close the create_service call

        self.get_logger().info('Waiting for MoveIt / controller …') # Print a log message indicating the node is waiting for external servers
        self._cart_client.wait_for_service(timeout_sec=30.0) # Block execution until the Cartesian planning service is available (up to 30s)
        self._traj_client.wait_for_server(timeout_sec=30.0) # Block execution until the trajectory action server is available (up to 30s)
        self.get_logger().info('MouthFeedingPlanner ready — call /feed_trigger.') # Print a log message indicating initialization is complete

    # Callbacks # Comment denoting the start of the callback functions section
    def _mouth_cb(self, msg: PointStamped): # Define the callback function that runs every time a mouth point is received
        self._latest_mouth = msg # Store the received PointStamped message into the class variable
        # Continuously visualise the mouth and the camera's forward axis in # Existing comment explaining the diagnostic visualization
        # world frame — this is the hand-eye-calibration diagnostic. # Existing comment explaining the diagnostic visualization
        # If the pink ball stays fixed on the user's mouth as the arm moves, # Existing comment explaining the diagnostic visualization
        # the camera-to-end-effector TF is correct.  If it drifts, the # Existing comment explaining the diagnostic visualization
        # cam_x/y/z/roll/pitch/yaw args in the launch file need tuning. # Existing comment explaining the diagnostic visualization
        self._publish_camera_diagnostic(msg) # Call the helper method to render diagnostic markers in RViz

    def _publish_camera_diagnostic(self, mouth_cam: PointStamped): # Define the helper method to publish diagnostic markers
        try: # Start a try-except block to catch potential TF transformation errors
            tf = self._tf_buf.lookup_transform( # Ask the TF buffer for the transform between two frames
                WORLD_FRAME, CAM_FRAME, # Specify the target frame (world) and source frame (camera)
                rclpy.time.Time(), # Request the most recent available transform (time 0)
                timeout=rclpy.duration.Duration(seconds=0.2), # Allow up to 0.2 seconds to wait for the transform to become available
            ) # Close the lookup_transform call
        except (LookupException, ExtrapolationException): # Catch exceptions if the transform isn't found or is too old/new
            return # Exit the function early if the transform fails

        mouth_w = do_transform_point(mouth_cam, tf) # Apply the retrieved transform to convert the mouth point to the world frame
        self._publish_sphere( # Call the helper method to draw a sphere at the transformed mouth coordinates
            mouth_w.point.x, mouth_w.point.y, mouth_w.point.z, # Pass the X, Y, and Z coordinates of the mouth in the world frame
            mid=0, rgba=(1.0, 0.4, 0.7, 0.9), diameter=0.05, # Pass marker ID 0, pinkish RGBA color, and a 5cm diameter
        ) # Close the _publish_sphere call

        # Arrow along camera +Z in world — shows where the lens points. # Existing comment explaining the arrow marker
        cam_origin = PointStamped() # Create a new PointStamped message to represent the camera's origin
        cam_origin.header.frame_id = CAM_FRAME # Set the frame of reference for the origin point to the camera frame
        cam_origin.point.x = cam_origin.point.y = cam_origin.point.z = 0.0 # Set the origin coordinates to exactly (0, 0, 0)
        cam_tip = PointStamped() # Create a new PointStamped message to represent the tip of the diagnostic arrow
        cam_tip.header.frame_id = CAM_FRAME # Set the frame of reference for the tip to the camera frame
        cam_tip.point.z = 0.20  # Set the Z coordinate to 0.20 meters (camera points along positive Z)
        p0 = do_transform_point(cam_origin, tf).point # Transform the camera origin to the world frame and extract the point geometry
        p1 = do_transform_point(cam_tip, tf).point # Transform the camera tip to the world frame and extract the point geometry

        arrow = Marker() # Create a new Marker message object for the arrow
        arrow.header.frame_id = WORLD_FRAME # Set the arrow's reference frame to the world frame
        arrow.header.stamp    = self.get_clock().now().to_msg() # Timestamp the marker with the current ROS time
        arrow.ns = 'feeding' # Set the marker namespace to 'feeding' to group it in RViz
        arrow.id = 2 # Assign a unique integer ID to this marker
        arrow.type   = Marker.ARROW # Set the marker shape type to ARROW
        arrow.action = Marker.ADD # Set the action to ADD (which creates or updates the marker)
        arrow.points = [Point(x=p0.x, y=p0.y, z=p0.z), # Define the start point of the arrow (p0)
                        Point(x=p1.x, y=p1.y, z=p1.z)] # Define the end point of the arrow (p1)
        arrow.scale.x = 0.01   # Set the arrow shaft diameter to 1 cm
        arrow.scale.y = 0.02   # Set the arrow head diameter to 2 cm
        arrow.scale.z = 0.03   # Set the arrow head length to 3 cm
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = ( # Start assigning the RGBA color values for the arrow
            0.2, 0.6, 1.0, 0.9 # Set the arrow color to a translucent blue
        ) # Close the color assignment
        arrow.pose.orientation.w = 1.0 # Initialize the quaternion's W component to 1.0 (valid unrotated quaternion)
        self._marker_pub.publish(arrow) # Publish the constructed arrow marker to the RViz topic

    def _js_cb(self, msg: JointState): # Define the callback function for incoming joint state messages
        self._latest_js = msg # Update the class variable with the most recently received joint state

    # ── Trigger ────────────────────────────────────────────────────────── # Comment denoting the section for the trigger service
    def _trigger_cb(self, _req, response): # Define the callback function that executes when the /feed_trigger service is called
        if self._busy: # Check if the robot is already executing a feeding motion
            response.success = False # Set the service response status to failure
            response.message = 'Feed already in progress.' # Set the failure message indicating the system is busy
            return response # Return the failed response to the caller
        if self._latest_mouth is None: # Check if a mouth position has been received yet
            response.success = False # Set the service response status to failure
            response.message = 'No mouth point yet — is mouth_tracker running?' # Set the failure message indicating missing data
            return response # Return the failed response to the caller

        self._busy = True # Lock the system by setting the busy flag to True
        try: # Start a try block to handle the feeding execution sequence safely
            ok, msg = self._execute_feed() # Call the main feeding logic method and capture its success boolean and message
            response.success = ok # Populate the service response success field with the execution result
            response.message = msg # Populate the service response message field with the execution message
        except Exception as exc: # Catch any unexpected Python exceptions during the execution
            response.success = False # Set the service response status to failure due to exception
            response.message = f'Exception: {exc}' # Format the exception message into the response string
            self.get_logger().error(response.message) # Log the error message to the ROS console
        finally: # Execute this block regardless of success or failure
            self._busy = False # Release the lock by setting the busy flag back to False
        return response # Return the populated service response back to the caller

    # ── Markers ────────────────────────────────────────────────────────── # Comment denoting the section for marker publishing helpers
    def _publish_sphere(self, x, y, z, mid, rgba, diameter=0.04): # Define a helper method to easily publish spherical markers
        m = Marker() # Create a new Marker message object
        m.header.frame_id = WORLD_FRAME # Set the sphere's reference frame to the world frame
        m.header.stamp    = self.get_clock().now().to_msg() # Timestamp the marker with the current ROS time
        m.ns              = 'feeding' # Assign the marker to the 'feeding' namespace
        m.id              = mid # Set the marker's unique integer ID (allows updating specific markers)
        m.type            = Marker.SPHERE # Set the marker shape type to SPHERE
        m.action          = Marker.ADD # Set the action to ADD (create or update)
        m.pose.position.x = float(x) # Cast and assign the X coordinate of the sphere
        m.pose.position.y = float(y) # Cast and assign the Y coordinate of the sphere
        m.pose.position.z = float(z) # Cast and assign the Z coordinate of the sphere
        m.pose.orientation.w = 1.0 # Set valid default quaternion orientation (w=1)
        m.scale.x = m.scale.y = m.scale.z = diameter # Set the X, Y, and Z scale to create a uniform sphere of the given diameter
        m.color.r, m.color.g, m.color.b, m.color.a = rgba # Unpack and assign the RGBA color tuple to the marker
        m.lifetime.sec = 0    # Set the marker lifetime to 0 seconds, meaning it persists forever until overwritten
        self._marker_pub.publish(m) # Publish the sphere marker to the visualization topic

    # ── Core ───────────────────────────────────────────────────────────── # Comment denoting the section for core logic
    def _execute_feed(self): # Define the main procedural method to calculate and execute the feeding motion
        # 1. Snapshot the mouth observation (decouple from the live topic) # Existing comment explaining snapping the mouth position
        snap = PointStamped() # Create a new PointStamped message to hold the static snapshot
        snap.header = self._latest_mouth.header # Copy the header (frame and timestamp) from the latest live mouth data
        snap.point  = self._latest_mouth.point # Copy the geometric point coordinates from the latest live mouth data

        # 2. mouth → world # Existing comment explaining transformation step
        try: # Start a try-except block to safely fetch the TF transform
            tf_cam = self._tf_buf.lookup_transform( # Ask the TF buffer for the transform
                WORLD_FRAME, CAM_FRAME, # Request the transform from the camera frame to the world frame
                rclpy.time.Time(), # Request the most recent available transform
                timeout=rclpy.duration.Duration(seconds=2.0), # Wait up to 2 seconds for the transform to become available
            ) # Close lookup_transform call
        except (LookupException, ExtrapolationException) as e: # Catch standard TF exceptions
            err = f'TF {CAM_FRAME}→{WORLD_FRAME} failed: {e}' # Format an error string with the exception details
            self.get_logger().error(err) # Log the TF error to the ROS console
            return False, err # Return a failure status and the error string

        mouth_w = do_transform_point(snap, tf_cam) # Transform the mouth snapshot from the camera frame into the world frame
        mx, my, mz = mouth_w.point.x, mouth_w.point.y, mouth_w.point.z # Extract the transformed X, Y, Z coordinates into local variables
        self.get_logger().info(f'Mouth  (world): ({mx:+.3f}, {my:+.3f}, {mz:+.3f}) m') # Log the calculated mouth world coordinates

        # Pink sphere = mouth position # Existing comment explaining the visual marker
        self._publish_sphere(mx, my, mz, mid=0, rgba=(1.0, 0.4, 0.7, 0.9), diameter=0.05) # Call the helper to publish a pink sphere at the mouth location

        # 3. Current EE pose in world # Existing comment explaining the fetch of the end-effector pose
        try: # Start try block to safely fetch the end-effector transform
            tf_ee = self._tf_buf.lookup_transform( # Ask the TF buffer for the transform
                WORLD_FRAME, EE_LINK, # Request the transform from the end-effector frame to the world frame
                rclpy.time.Time(), # Request the most recent available transform
                timeout=rclpy.duration.Duration(seconds=2.0), # Wait up to 2 seconds for the transform to become available
            ) # Close lookup_transform call
        except (LookupException, ExtrapolationException) as e: # Catch standard TF exceptions
            err = f'TF {EE_LINK}→{WORLD_FRAME} failed: {e}' # Format an error string with the exception details
            self.get_logger().error(err) # Log the TF error to the console
            return False, err # Return a failure status and the error string

        ex = tf_ee.transform.translation.x # Extract the end-effector current X position from the transform
        ey = tf_ee.transform.translation.y # Extract the end-effector current Y position from the transform
        ez = tf_ee.transform.translation.z # Extract the end-effector current Z position from the transform
        cur_ori = tf_ee.transform.rotation # Extract the end-effector current orientation (quaternion) from the transform
        self.get_logger().info(f'EE     (world): ({ex:+.3f}, {ey:+.3f}, {ez:+.3f}) m') # Log the current end-effector position

        # 4. EE → mouth direction + distance # Existing comment explaining distance and vector calculation
        dx, dy, dz = mx - ex, my - ey, mz - ez # Calculate the delta vector components from the end-effector to the mouth
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) # Calculate the straight-line Euclidean distance between EE and mouth
        if dist < 1e-3: # Check if the distance is extremely small (less than 1 mm), avoiding division by zero later
            return False, f'Mouth and EE coincide (dist={dist:.4f} m).' # Abort if the end-effector is exactly at the mouth
        if dist <= TOTAL_OFFSET_M: # Check if the arm is already within the designated standoff distance
            return False, (f'Already within {TOTAL_OFFSET_M*100:.0f} cm of mouth ' # Format the abort message (part 1)
                           f'(dist={dist*100:.1f} cm) — nothing to do.') # Format the abort message (part 2) and return failure

        ux, uy, uz = dx / dist, dy / dist, dz / dist # Calculate the unit vector (direction) from the end-effector to the mouth
        self.get_logger().info(f'EE→mouth dist: {dist*100:.1f} cm   dir: ' # Log the calculated distance and direction vector (part 1)
                               f'({ux:+.2f}, {uy:+.2f}, {uz:+.2f})') # Log the calculated distance and direction vector (part 2)

        # 5. Target = mouth − 10 cm along direction  (spoon tip 5 cm from mouth) # Existing comment explaining target calculation
        tx = mx - TOTAL_OFFSET_M * ux # Calculate target X: start at mouth X and pull back by TOTAL_OFFSET_M along the direction vector
        ty = my - TOTAL_OFFSET_M * uy # Calculate target Y: start at mouth Y and pull back by TOTAL_OFFSET_M along the direction vector
        tz = mz - TOTAL_OFFSET_M * uz # Calculate target Z: start at mouth Z and pull back by TOTAL_OFFSET_M along the direction vector

        # Clamp to workspace: the j2s6s200 can only reach ~0.85 m from its # Existing comment explaining workspace limitations
        # base.  If the target is farther, cap it along EE→mouth so the arm # Existing comment explaining clamping behavior
        # at least moves in the right direction (and stops short). # Existing comment explaining clamping behavior
        t_norm = math.sqrt(tx * tx + ty * ty + tz * tz) # Calculate the distance from the robot base (world origin) to the target
        if t_norm > MAX_REACH: # Check if the calculated target is outside the robot's physical reach sphere
            scale = MAX_REACH / t_norm # Calculate a scaling factor to bring the target back onto the edge of the reach sphere
            tx, ty, tz = tx * scale, ty * scale, tz * scale # Apply the scaling factor to the target coordinates
            self.get_logger().warn( # Output a warning log that clamping occurred
                f'Target {t_norm:.2f} m from base > MAX_REACH {MAX_REACH:.2f} m — ' # Log original distance vs limit
                f'clamped to ({tx:+.3f}, {ty:+.3f}, {tz:+.3f}).' # Log the new clamped coordinates
            ) # Close warning log call

        self.get_logger().info(f'Target (world): ({tx:+.3f}, {ty:+.3f}, {tz:+.3f}) m ' # Log the final computed target position
                               f'(travel {dist - TOTAL_OFFSET_M:.3f} m)') # Log the expected travel distance

        # Green sphere = commanded target (spoon tip location approx) # Existing comment explaining the target visualization
        self._publish_sphere(tx, ty, tz, mid=1, rgba=(0.2, 1.0, 0.2, 0.9), diameter=0.04) # Publish a green sphere marker at the target position

        # 6. Preserve current orientation — no rotation, pure translation # Existing comment explaining the orientation strategy
        target = Pose( # Create a geometry_msgs/Pose object for the target waypoint
            position=Point(x=float(tx), y=float(ty), z=float(tz)), # Populate the translation part with the computed target coordinates
            orientation=Quaternion( # Populate the rotation part with the exact current orientation of the end-effector
                x=cur_ori.x, y=cur_ori.y, z=cur_ori.z, w=cur_ori.w, # Copy the quaternion components
            ), # Close Quaternion initialization
        ) # Close Pose initialization

        # 7. Plan & execute a straight-line Cartesian move # Existing comment explaining the path planning step
        traj = self._cartesian_path([target]) # Call the helper function to compute a straight Cartesian path to the target pose
        if traj is None or not traj.joint_trajectory.points: # Check if the path planning failed or returned an empty trajectory
            return False, 'Cartesian planning returned no trajectory.' # Return failure and an error message

        ok = self._exec_trajectory(traj) # Call the helper function to send the trajectory to the robot controller and wait for execution
        return ok, ('Feed complete.' if ok else 'Trajectory execution failed.') # Return the execution status and appropriate message

    # ── Cartesian planning ─────────────────────────────────────────────── # Comment denoting the section for path planning
    def _cartesian_path(self, waypoints): # Define a method to call the MoveIt ComputeCartesianPath service
        req = GetCartesianPath.Request() # Create a new request object for the service
        req.header.frame_id  = WORLD_FRAME # Set the reference frame for the waypoints in the request
        req.group_name       = ARM_GROUP # Specify which MoveIt planning group to use (the arm)
        req.link_name        = EE_LINK # Specify which robot link should follow the path (end effector)
        req.waypoints        = waypoints # Attach the list of target poses (waypoints) to the request
        req.max_step         = CART_STEP # Set the maximum distance between interpolation points on the path
        req.jump_threshold   = 0.0 # Set the jump threshold to 0.0 to disable configuration jump detection (can be risky, but default here)
        req.avoid_collisions = False   # Set to false to ignore collision objects in the environment (simplifies planning for testing)

        if self._latest_js is not None: # Check if we have received a joint state to use as the starting point
            rs = RobotState() # Create a new RobotState message object
            rs.joint_state = self._latest_js # Populate it with the latest known joint states
            req.start_state = rs # Set this as the explicit start state for the motion plan

        future = self._cart_client.call_async(req) # Send the Cartesian path planning request asynchronously
        rclpy.spin_until_future_complete(self, future) # Block the current thread until the service responds
        res = future.result() # Extract the result object from the completed future
        self.get_logger().info(f'Cartesian coverage: {res.fraction * 100:.1f} %') # Log the percentage of the requested path that was successfully planned
        return res.solution # Return the resulting RobotTrajectory message

    # ── Trajectory execution ───────────────────────────────────────────── # Comment denoting the section for trajectory execution
    def _add_timestamps(self, robot_traj, speed=FEED_SPEED): # Define a method to calculate and add time-from-start values to trajectory points
        pts = robot_traj.joint_trajectory.points # Extract the list of trajectory points
        if not pts: # Check if the list of points is empty
            return robot_traj # Return the empty trajectory unmodified
        pts[0].time_from_start = Duration(sec=0, nanosec=0) # Set the time for the very first point to zero
        total, prev = 0.0, pts[0] # Initialize a running time total and a reference to the previous point
        for pt in pts[1:]: # Iterate through all trajectory points starting from the second one
            delta = ( # Start calculation of maximum joint angle change
                max(abs(a - b) for a, b in zip(pt.positions, prev.positions)) # Find the largest angular difference across all joints between prev and current point
                if prev.positions and pt.positions else 0.01 # Fallback to a small delta if position arrays are empty
            ) # Close delta calculation
            total += max(delta / speed, MIN_DT) # Calculate time needed for this step based on speed, bounded by MIN_DT, and add to total
            s  = int(total) # Extract the whole seconds part of the total time
            ns = int((total - s) * 1e9) # Calculate the remaining fractional seconds and convert to nanoseconds
            pt.time_from_start = Duration(sec=s, nanosec=ns) # Assign the calculated duration object to the current point
            prev = pt # Update the previous point reference to the current one for the next iteration
        robot_traj.joint_trajectory.points = pts # Reassign the updated points list back to the trajectory object
        return robot_traj # Return the fully timed trajectory

    def _exec_trajectory(self, robot_traj) -> bool: # Define a method to send a trajectory to the action server and wait for completion
        robot_traj = self._add_timestamps(robot_traj) # Call the helper to apply timing information to the raw trajectory
        goal = FollowJointTrajectory.Goal() # Create a new goal object for the FollowJointTrajectory action
        goal.trajectory = robot_traj.joint_trajectory # Assign the timed joint trajectory to the goal

        future = self._traj_client.send_goal_async(goal) # Send the goal asynchronously to the action server
        rclpy.spin_until_future_complete(self, future) # Block the current thread until the action server accepts or rejects the goal
        gh = future.result() # Extract the goal handle from the completed future
        if not gh.accepted: # Check if the action server rejected the trajectory goal
            self.get_logger().error('Trajectory rejected by controller.') # Log an error if rejected
            return False # Return failure

        res_f = gh.get_result_async() # Request the final execution result asynchronously from the goal handle
        rclpy.spin_until_future_complete(self, res_f) # Block the thread until the arm finishes moving and the result is returned
        code = res_f.result().result.error_code # Extract the specific integer error code from the action result
        self.get_logger().info(f'Controller error_code: {code}') # Log the numerical error code (0 means SUCCESS)
        return code == 0 # Return True if the error code is 0 (success), otherwise False


def main(args=None): # Define the main entry point function for the script
    rclpy.init(args=args) # Initialize the ROS 2 Python client library infrastructure
    node = MouthFeedingPlanner() # Instantiate the MouthFeedingPlanner ROS node
    executor = MultiThreadedExecutor() # Instantiate a MultiThreadedExecutor to allow callbacks to run concurrently
    executor.add_node(node) # Attach our custom node to the executor
    try: # Start a try block to handle graceful shutdown
        executor.spin() # Start the executor loop, which blocks and processes incoming ROS messages/service calls
    finally: # Execute this cleanup block when spin() exits (e.g., via Ctrl+C)
        node.destroy_node() # Cleanly destroy the node and its associated ROS entities
        rclpy.shutdown() # Shutdown the ROS 2 Python client library context


if __name__ == '__main__': # Standard Python idiom to check if the script is being executed directly (not imported)
    main() # Call the main function to start the program