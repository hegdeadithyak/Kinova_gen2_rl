import pyrealsense2 as rs

# Start pipeline and get profile
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(config)

# Get intrinsics from the depth stream
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_stream.get_intrinsics()

# Extract values
fx = intrinsics.fx
fy = intrinsics.fy
cx = intrinsics.ppx  # SDK uses ppx for cx
cy = intrinsics.ppy  # SDK uses ppy for cy

print(f"fx: {fx}, fy: {fy}, cx (ppx): {cx}, cy (ppy): {cy}")   
