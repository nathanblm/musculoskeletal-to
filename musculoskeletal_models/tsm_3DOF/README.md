# TSM 3-DOF model

`tsm_3DOF.osim` is adapted from `TSM_subject_noWeight.osim` for three-coordinate
map generation and RMR optimization:

- Free coordinates: `plane_elv`, `shoulder_elv`, and `elbow_flexion`.
- `axial_rot` is locked at its default value of `1.0288316440818757` rad, and
  its coordinate actuator is removed.
- `ground_thorax`, `sternoclavicular`, `scapulothoracic`, and `radioulnar` are
  welds preserving their original default poses. The four scapula coordinates,
  fixed-joint coordinate actuators, and `AC` point constraint are removed.
- A `ground_offset` frame raises the model by `0.8 m` along ground Y, matching
  the arm26 model base height.
- Added `hand_external_force` at the hand body origin. Force components are in
  the ground frame and applied torque is zero.
- Added the `endpoint` marker at the `/bodyset/hand` origin: `(0, 0, 0) m`.

