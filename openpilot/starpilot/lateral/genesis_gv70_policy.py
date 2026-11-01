"""First-generation Electrified GV70 torque shaping."""

from openpilot.starpilot.flm import torque_stages as auto_stages
import math
from collections import deque

import numpy as np

from openpilot.starpilot.lateral.torque_extension import apply_turn_assist
from openpilot.cereal import log
from opendbc.car import structs
from opendbc.car.hyundai.values import CAR
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from openpilot.starpilot.lateral.torque_shaping import (
  INTERP_SPEEDS,
  KP_INTERP,
  KI,
  LAT_ACCEL_REQUEST_BUFFER_SECONDS,
  LP_FILTER_CUTOFF_HZ,
  MAX_LAT_JERK_UP,
  LOW_SPEED_X,
  LOW_SPEED_Y,
  MIN_SPEED,
  JERK_GAIN,
  FF_ROLL_OFFSET_FADE_BP,
  FF_ROLL_OFFSET_FADE_V,
  UNWIND_D_DES_THRESHOLD,
  UNWIND_LAT_ACCEL_NEAR_ZERO,
  center_chatter_friction_jerk_deadzone,
)


from openpilot.starpilot.lateral.gv70_shaping import (
  get_genesis_gv70_friction_threshold,
  get_genesis_gv70_friction_jerk_deadzone,
  get_genesis_gv70_center_output_scale,
  get_genesis_gv70_unwind_ff_scale,
  get_genesis_gv70_high_speed_error_scale,
  get_genesis_gv70_reversal_output_scale,
  get_genesis_gv70_low_speed_center_overshoot_scale,
  get_genesis_gv70_stabilized_output,
  GENESIS_GV70_MEASUREMENT_DAMPING_SPEED_BP, GENESIS_GV70_MEASUREMENT_DAMPING_V,
)

def supported_cp(CP):
  return (
    CP.brand == "hyundai"
    and CP.carFingerprint == CAR.GENESIS_GV70_ELECTRIFIED_1ST_GEN
    and CP.steerControlType == structs.CarParams.SteerControlType.torque
    and CP.lateralTuning.which() == "torque"
    and not CP.dashcamOnly
    and not CP.passive
    and not CP.notCar
  )



class GenesisGV70TorquePolicy:
  def __init__(self, parent, CP):
    if not supported_cp(CP):
      raise ValueError("Electrified GV70 policy does not match CarParams")
    self.parent = parent
    self.dt = parent.dt
    parent.pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI,
                               [GENESIS_GV70_MEASUREMENT_DAMPING_SPEED_BP, GENESIS_GV70_MEASUREMENT_DAMPING_V],
                               rate=1 / self.dt)
    parent.update_limits()
    self.request_buffer_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / self.dt)
    self.curvature_request_buffer = deque([0.0] * self.request_buffer_len, maxlen=self.request_buffer_len)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), self.dt)
    self.measurement_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * (MAX_LAT_JERK_UP - 0.5)), self.dt)
    self.previous_measurement = 0.0
    self.prev_desired_lateral_accel = 0.0
    self.prev_steering_pressed = False
    self.prev_output_torque = 0.0
    self.previous_feedforward = None
    self.low_speed_reset_threshold = max(CP.minSteerSpeed, 0.3)
    # Approach-build state (gv70-experimental only, precision over comfort).
    # 50 frames = ~0.5 s at 100 Hz; boost is a magnitude in lateral-accel
    # units, ramped at 0.05/frame, and is added to feedforward only.
    self.approach_curv_hist = deque([0.0] * 50, maxlen=50)
    self.approach_boost = 0.0

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    parent = self.parent
    pid_log = log.ControlsState.LateralTorqueState.new_message()
    pid_log.version = 2
    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    measurement = measured_curvature * CS.vEgo**2
    future_desired_lateral_accel = desired_curvature * CS.vEgo**2
    if not active:
      parent.pid.reset()
      self.prev_output_torque = 0.0
      self.previous_feedforward = None
      self.approach_boost = 0.0
      self.approach_curv_hist.append(desired_curvature)
      self.curvature_request_buffer.append(desired_curvature)
      self.previous_measurement = measurement
      self.measurement_rate_filter.x = 0.0
      self.jerk_filter.x = 0.0
      self.prev_desired_lateral_accel = future_desired_lateral_accel
      self.prev_steering_pressed = CS.steeringPressed
      pid_log.active = False
      return 0.0, 0.0, pid_log

    if self.prev_steering_pressed and not CS.steeringPressed:
      parent.pid.i *= 0.8
    roll_offset_fade = float(np.interp(CS.vEgo, FF_ROLL_OFFSET_FADE_BP, FF_ROLL_OFFSET_FADE_V))
    roll_compensation = params.roll * ACCELERATION_DUE_TO_GRAVITY * roll_offset_fade
    curvature_deadzone = abs(VM.calc_curvature(math.radians(parent.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    lateral_accel_deadzone = curvature_deadzone * CS.vEgo**2
    lateral_accel_deadzone = auto_stages.apply_selected_deadzone(parent, CS, VM, float(lateral_accel_deadzone), desired_curvature,
      active=active, limited=steer_limited_by_safety or curvature_limited, curvature_limited=curvature_limited)
    delay_frames = int(np.clip(lat_delay / self.dt, 1, self.request_buffer_len))
    expected_lateral_accel = self.curvature_request_buffer[-delay_frames] * CS.vEgo**2
    self.curvature_request_buffer.append(desired_curvature)
    raw_lateral_jerk = (future_desired_lateral_accel - expected_lateral_accel) / max(lat_delay, self.dt)
    raw_lateral_jerk = float(np.clip(raw_lateral_jerk, -MAX_LAT_JERK_UP, MAX_LAT_JERK_UP))
    desired_lateral_jerk = float(np.clip(self.jerk_filter.update(raw_lateral_jerk), -MAX_LAT_JERK_UP, MAX_LAT_JERK_UP))
    gravity_adjusted_future_lateral_accel = future_desired_lateral_accel - roll_compensation
    setpoint = expected_lateral_accel + desired_lateral_jerk * lat_delay
    desired_lateral_accel_magnitude_rate = (abs(setpoint) - abs(self.prev_desired_lateral_accel)) / self.dt
    unwind_detected = desired_lateral_accel_magnitude_rate < UNWIND_D_DES_THRESHOLD and abs(setpoint) < UNWIND_LAT_ACCEL_NEAR_ZERO
    self.prev_desired_lateral_accel = setpoint
    measurement_rate = self.measurement_rate_filter.update((measurement - self.previous_measurement) / self.dt)
    measurement_rate = float(np.clip(measurement_rate, -MAX_LAT_JERK_UP, MAX_LAT_JERK_UP))
    self.previous_measurement = measurement
    low_speed_factor = (np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y) / max(CS.vEgo, MIN_SPEED)) ** 2
    current_kp = np.interp(CS.vEgo, parent.pid._k_p[0], parent.pid._k_p[1])
    error = setpoint - measurement
    error_with_lsf = error * (1 + low_speed_factor / max(current_kp, 1e-3))
    pid_log.error = float(error_with_lsf)
    ff = gravity_adjusted_future_lateral_accel - parent.torque_params.latAccelOffset * roll_offset_fade
    ff *= get_genesis_gv70_unwind_ff_scale(setpoint, measurement, desired_lateral_jerk, CS.vEgo)
    friction_threshold = get_genesis_gv70_friction_threshold(CS.vEgo, setpoint, desired_lateral_jerk)
    vehicle_deadzone = get_genesis_gv70_friction_jerk_deadzone(CS.vEgo, setpoint)
    friction_jerk_deadzone = center_chatter_friction_jerk_deadzone(CS.vEgo, setpoint, vehicle_deadzone)
    friction_jerk = math.copysign(max(abs(desired_lateral_jerk) - friction_jerk_deadzone, 0.0), desired_lateral_jerk)
    ff, friction_threshold = auto_stages.apply_stages(parent, CS, setpoint, desired_lateral_jerk, ff, friction_threshold,
      active=active, limited=steer_limited_by_safety or curvature_limited, curvature_limited=curvature_limited)
    selected_friction = parent.friction(error_with_lsf + JERK_GAIN * friction_jerk, lateral_accel_deadzone, friction_threshold, CS, setpoint)
    ff += auto_stages.apply_selected_friction(parent, CS, setpoint, float(selected_friction),
      active=active, limited=steer_limited_by_safety or curvature_limited, curvature_limited=curvature_limited)
    if CS.vEgo < self.low_speed_reset_threshold:
      parent.pid.reset()
    # Filter feedforward before PID so feedback remains immediate.
    ff *= get_genesis_gv70_center_output_scale(setpoint, CS.vEgo)
    ff *= get_genesis_gv70_low_speed_center_overshoot_scale(setpoint, measurement, CS.vEgo)
    ff *= get_genesis_gv70_high_speed_error_scale(setpoint, measurement, desired_lateral_jerk, CS.vEgo)
    ff *= get_genesis_gv70_reversal_output_scale(setpoint, measurement, desired_lateral_jerk, CS.vEgo)
    if not CS.steeringPressed and not self.prev_steering_pressed and self.previous_feedforward is not None:
      ff = get_genesis_gv70_stabilized_output(ff, self.previous_feedforward, setpoint,
                                            desired_lateral_jerk, CS.vEgo, self.dt)
    # --- Approach-phase build (gv70-experimental only) ---
    # Target: build steering angle EARLIER, while the approach is only
    # 18-52% saturated and headroom exists, so the wheel is further along
    # when demand peaks and the (already saturated) core can maintain it.
    # This cannot raise max EPS or max command (both still capped at
    # 1.0 / 409 by PID limits + carcontroller); it only moves command
    # earlier in a rising-curvature approach.
    # Gates (all must hold for target > 0):
    #  * curvature rising for ~0.5 s, same sign, no sign flip >0.008
    #    in that window (stable path, not a transient / lane change step)
    #  * |desired_curvature| >= 0.003 (R <= ~333 m; Route 2 BM1 peaks
    #    at only 0.0071, so a 0.008 floor would never fire there, while
    #    straight driving sits at ~0.0001, so 0.003 is still well
    #    above straight noise)
    #  * 0.5 s curvature trend >= 0.001 (BM1 trends 0.0010-0.0015,
    #    BM2 0.0019-0.0045, BM3 0.0030-0.0061)
    #  * speed >= 8 m/s, full above 12 m/s (Route 2 curves: 9-15 m/s)
    #  * setpoint in the approach zone 0.3-1.9 latAccel, full at 1.0,
    #    fading to 0 at the 1.9 cap (no effect in saturated cores)
    #  * driver not pressed, not safety/curvature limited
    # NOT gated on lane confidence: LatControlTorque.update receives
    # no model lane-confidence signal, so a low-confidence curve
    # (Route 2 BM3, 0.13/0.21) cannot be excluded here by code. The
    # test protocol must therefore exclude low-confidence curves, and
    # a future controlsd-level gate would be needed for a PR version.
    # Negative consequences, mitigated by the gates + cap + ramp:
    #  * steering into a curve early on a wrong model path -> cuts
    #    the corner / drifts inside (mitigated: sign-stable 0.5 s
    #    trend, curvature threshold, capped boost, ramped in/out)
    #  * overshoot / oscillation on unwind (mitigated: boost fades
    #    to 0 as setpoint approaches the cap and ramps down at
    #    0.05/frame when the gate closes, not a step)
    #  * false trigger on a lane change / demand spike (mitigated:
    #    0.5 s sign-stable trend, not a single-frame jerk spike)
    #  * abrupt feel at high speed (mitigated: cap 0.45 latAccel
    #    ~= 0.24 normalised, ramped, not a step to full command)
    #  * if the demand peak still exceeds the cap, this only delays
    #    failure, it does not fix it (stated, not hidden).
    self.approach_curv_hist.append(desired_curvature)
    _trend = desired_curvature - self.approach_curv_hist[0] if len(self.approach_curv_hist) >= 50 else 0.0
    _sign_stable = not any(v * desired_curvature < 0.0 and abs(v) > 0.008 for v in self.approach_curv_hist)
    _rising = (_trend * desired_curvature > 0.0 and abs(_trend) >= 0.001
               and abs(desired_curvature) >= 0.003 and _sign_stable)
    _target_boost = 0.0
    if (_rising and not CS.steeringPressed and not steer_limited_by_safety
        and not curvature_limited and setpoint != 0.0):
      _speed_w = float(np.interp(CS.vEgo, [8.0, 12.0], [0.0, 1.0]))
      _mag_w = float(np.interp(abs(setpoint), [0.3, 1.0, 1.9], [0.0, 1.0, 0.0]))
      _target_boost = 0.45 * _speed_w * _mag_w
    _boost_step = float(np.clip(_target_boost - self.approach_boost, -0.05, 0.05))
    self.approach_boost += _boost_step
    if self.approach_boost != 0.0 and setpoint != 0.0:
      ff += math.copysign(self.approach_boost, setpoint)
    self.previous_feedforward = ff
    error_rate = 0.0 if CS.steeringPressed else -measurement_rate
    freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < self.low_speed_reset_threshold or unwind_detected
    output_lataccel = parent.pid.update(pid_log.error, error_rate=error_rate, speed=CS.vEgo, feedforward=ff, freeze_integrator=freeze_integrator)
    output_torque = parent.torque_from_lateral_accel(output_lataccel, parent.torque_params)
    output_torque = apply_turn_assist(parent, CS, VM, params, desired_curvature, output_torque)
    pid_log.active = True
    pid_log.p = float(parent.pid.p)
    pid_log.i = float(parent.pid.i)
    pid_log.d = float(parent.pid.d)
    pid_log.f = float(parent.pid.f)
    pid_log.output = float(-output_torque)
    pid_log.actualLateralAccel = float(measurement)
    pid_log.desiredLateralAccel = float(setpoint)
    pid_log.desiredLateralJerk = float(desired_lateral_jerk)
    pid_log.saturated = bool(parent._check_saturation(parent.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))
    self.prev_steering_pressed = CS.steeringPressed
    self.prev_output_torque = float(output_torque)
    return -output_torque, 0.0, pid_log
