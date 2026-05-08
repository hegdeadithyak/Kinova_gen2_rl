#!/bin/bash

echo "Setting use_sim_time=true for all active ROS 2 nodes..."

# Check if ROS 2 is available
if ! command -v ros2 &>/dev/null; then
  echo "ROS 2 command not found. Did you source your setup.bash?"
  exit 1
fi

# Get list of nodes
nodes=$(ros2 node list)

if [ -z "$nodes" ]; then
  echo "No active ROS 2 nodes found."
  exit 0
fi

# Loop through each node
for node in $nodes; do
  echo "Setting use_sim_time for $node"
  ros2 param set "$node" use_sim_time true
done

echo "Done."
