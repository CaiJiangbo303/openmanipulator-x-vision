# Semantic Tracking and Recovery

## Features

- Semantic-compatible association: blue block | plastic cube.
- Conservative lost-track recovery.
- Track identity and state generation separation.
- Independent tracking recovery state machine.
- Fresh source and mask evidence validation.
- Recovery timeout and ambiguity rejection.
- Read-only shadow authorization.

## Experimental configuration

- Acquire confidence: 0.50
- Tracking confidence: 0.40
- Mask confidence: 0.40
- Short hold: 0.35 s
- Recovery timeout: 3.0 s
- Recovery position delta: 0.030 m
- Required independent updates: 3
- Minimum validation duration: 0.20 s

Enable tracker occlusion recovery explicitly:

    -p enable_occlusion_recovery:=true
    -p lost_retention_sec:=3.0

## Validation results

- Semantic tracker synthetic tests: 5/5 PASS.
- Recovery manager unit tests: 11/11 PASS.
- Modular short occlusion recovery: PASS.
- Observed disappearance: 1.91 s.
- Reappearance-to-reauthorization: approximately 340 ms.
- Recovery position consistency: 0.18 mm.
- Recovery timeout: PASS.
- Raw candidate ambiguity rejection: PASS.
- Ambiguity test reauthorizations: 0.

## Tests

    python3 -B run_nodes/semantic_tracking/test_recovery_manager.py

    ROS_DOMAIN_ID=198 python3 -B \
        run_nodes/semantic_tracking/test_semantic_tracking.py

With the camera-only perception pipeline running:

    RECOVERY_SCENARIO=normal python3 -B \
        run_nodes/semantic_tracking/visual_gate_module.py

Available scenarios: normal, timeout, ambiguity.

## Limitations

Experimental shadow authorization only.
No motion commands are generated.
Position consistency is not absolute positioning accuracy.
Raw candidate counting is a conservative single-target policy.
Undetected objects and visually identical replacements remain risks.
Positive recovery needs regression after the raw candidate guard change.
Full multi-object association and hardware safety remain unvalidated.
