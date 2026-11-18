uint8 UNREACHABLE=0
uint8 MARGINAL=1
uint8 REACHABLE=2

string track_id
string class_id

float32 horizon

geometry_msgs/Point target_world
geometry_msgs/Point target_arm

float64[4] joint_positions
float64[4] joint_margins

float32 position_residual
float32 min_joint_margin

uint8 status
string reason

bool frozen_base_assumption
