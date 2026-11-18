uint8 TENTATIVE=0
uint8 TRACKED=1
uint8 LOST=2

string track_id
string class_id

geometry_msgs/Point position
geometry_msgs/Vector3 velocity

geometry_msgs/Quaternion orientation
bool orientation_observable

geometry_msgs/Vector3 position_std

float32 confidence

builtin_interfaces/Time last_seen

uint32 age
uint32 missed

uint8 state

string measurement_source
