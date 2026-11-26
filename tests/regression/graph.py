import json
import os
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.action.graph import get_action_names_and_types
rclpy.init(args=[])
name='graph_check_'+uuid.uuid4().hex[:10]
node=Node(name)
try:
    until=time.monotonic()+1.
    while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.04)
    peers=[v for v in node.get_node_names_and_namespaces() if v!=(name,'/')]
    actions=get_action_names_and_types(node)
    print(json.dumps(dict(domain=os.environ['ROS_DOMAIN_ID'],localhost=os.environ['ROS_LOCALHOST_ONLY'],nodes=peers,actions=actions)))
    assert not peers and not actions,'unexpected graph endpoints'
finally:
    node.destroy_node();rclpy.shutdown()
