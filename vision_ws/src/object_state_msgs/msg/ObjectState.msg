uint8 TENTATIVE=0
uint8 TRACKED=1
uint8 LOST=2

# Measurement provenance.
uint8 SOURCE_UNKNOWN=0
uint8 SOURCE_FAST_BBOX_DEPTH=1
uint8 SOURCE_PRECISE_MASK_DEPTH=2
uint8 SOURCE_FUSED=3

string track_id
# Unique identity of this state lifecycle.
string generation_id
string class_id

geometry_msgs/Point position
geometry_msgs/Vector3 velocity

geometry_msgs/Quaternion orientation
bool orientation_observable

geometry_msgs/Vector3 position_std

float32 confidence

builtin_interfaces/Time last_seen

# Timestamp of the source measurement underlying this state.
builtin_interfaces/Time source_stamp

# Age of source measurement when this state was published.
float32 measurement_age_ms

# Machine-readable provenance. Keep measurement_source string below
# for compatibility and human-readable diagnostics.
uint8 measurement_source_type

# True when published position has been propagated beyond source_stamp.
bool position_predicted

# Safe downstream validity / freshness flags.
bool valid_3d
bool stale

uint32 age
uint32 missed

uint8 state

string measurement_source
