import math
from unittest.mock import patch

import pytest

from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import log
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD

BASE = 10_000_000_000
SERVICES = ('carControl', 'carOutput', 'modelV2')


def gv70_params(vehicle=CAR.GENESIS_GV70_ELECTRIFIED_1ST_GEN):
  return interfaces[vehicle].get_non_essential_params(vehicle)


class Inputs(dict):
  def __init__(self):
    super().__init__(carControl=car.CarControl.new_message(latActive=True),
                     carOutput=car.CarOutput.new_message(),
                     modelV2=log.ModelDataV2.new_message())
    self['carOutput'].actuatorsOutput.torque = 1.0
    self.logMonoTime = {service: BASE for service in SERVICES}
    self.seen = {service: True for service in SERVICES}
    self.valid = {service: True for service in SERVICES}
    self.alive = {service: True for service in SERVICES}


def warning_state(monkeypatch):
  sd = SelfdriveD.__new__(SelfdriveD)
  sd.gv70_torque_steering = True
  sd.gv70_torque_saturation_since_ns = sd.gv70_torque_saturation_last_ns = 0
  sd.gv70_vehicle_model = VehicleModel(gv70_params().as_reader())
  sd.conditional_car_state_valid = True
  monkeypatch.setattr(sd, 'sm', Inputs(), raising=False)
  sd.events = Events()
  cs = car.CarState.new_message(canValid=True, gearShifter='drive')
  cs.vEgo = cs.vEgoRaw = 9.0
  return sd, cs


def desired_angle(sd, cs, curvature):
  return math.degrees(sd.gv70_vehicle_model.get_steer_from_curvature(-curvature, cs.vEgo, 0.0))


def sample(sd, cs, elapsed, *, torque=1.0, curvature=-0.0212, error=30.0, ages=None):
  now = BASE + elapsed
  ages = ages or {}
  for service in SERVICES:
    sd.sm.logMonoTime[service] = now - ages.get(service, 0)
  sd.aol_car_state_log_ns = now - ages.get('carState', 0)
  sd.sm['carOutput'].actuatorsOutput.torque = torque
  sd.sm['modelV2'].action.desiredCurvature = curvature
  cs.steeringAngleDeg = desired_angle(sd, cs, curvature) - math.copysign(error, desired_angle(sd, cs, curvature))
  sd.events.clear()
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=now):
    sd._update_gv70_torque_saturation_warning(cs)
  return log.OnroadEvent.EventName.steerSaturated in sd.events.names


@pytest.mark.parametrize('speed', (0., 2.2, 9.0))
@pytest.mark.parametrize('sign', (-1, 1))
def test_sustained_pin_with_large_error_warns_at_any_speed(speed, sign, monkeypatch):
  sd, cs = warning_state(monkeypatch)
  cs.vEgo = cs.vEgoRaw = speed
  for elapsed in range(0, 1_000_000_000, 10_000_000):
    assert not sample(sd, cs, elapsed, curvature=sign * -0.0212)
  assert not sample(sd, cs, 999_000_000, curvature=sign * -0.0212)
  assert sample(sd, cs, 1_000_000_000, curvature=sign * -0.0212)
  assert set(sd.events.event_rules[log.OnroadEvent.EventName.steerSaturated]) == {ET.WARNING}


def test_pinned_with_small_error_never_warns(monkeypatch):
  # A successful hard turn: output at the cap, wheel tracking its target.
  sd, cs = warning_state(monkeypatch)
  for elapsed in range(0, 2_100_000_000, 10_000_000):
    assert not sample(sd, cs, elapsed, error=3.0)


def test_large_error_without_pin_never_warns(monkeypatch):
  sd, cs = warning_state(monkeypatch)
  for elapsed in range(0, 2_100_000_000, 10_000_000):
    assert not sample(sd, cs, elapsed, torque=0.5, error=40.0)


def test_brief_pin_resets_confirmation(monkeypatch):
  sd, cs = warning_state(monkeypatch)
  for elapsed in range(0, 600_000_000, 10_000_000):
    assert not sample(sd, cs, elapsed)
  assert not sample(sd, cs, 600_000_000, torque=0.5)
  assert sd.gv70_torque_saturation_since_ns == 0
  for elapsed in range(610_000_000, 1_610_000_000, 10_000_000):
    assert not sample(sd, cs, elapsed)
  assert sample(sd, cs, 1_610_000_000)


@pytest.mark.parametrize('fault', ('driver', 'temporary_eps', 'permanent_eps', 'can', 'timeout', 'inactive',
  'invalid_source', 'low_torque', 'small_error', 'invalid_output', 'invalid_model', 'unseen_output',
  'dead_model', 'old_control', 'old_output', 'old_model', 'old_state', 'future_control',
  'nan_torque', 'nan_curvature', 'nan_angle'))
def test_invalid_or_withdrawn_input_resets_confirmation(fault, monkeypatch):
  sd, cs = warning_state(monkeypatch)
  for elapsed in range(0, 1_010_000_000, 10_000_000):
    sample(sd, cs, elapsed)
  assert log.OnroadEvent.EventName.steerSaturated in sd.events.names
  fields = {'driver': ('steeringPressed', True), 'temporary_eps': ('steerFaultTemporary', True),
            'permanent_eps': ('steerFaultPermanent', True), 'can': ('canValid', False),
            'timeout': ('canTimeout', True)}
  if fault in fields:
    setattr(cs, *fields[fault])
  sd.sm['carControl'].latActive = fault != 'inactive'
  sd.conditional_car_state_valid = fault != 'invalid_source'
  sd.sm.valid['carOutput'] = fault != 'invalid_output'
  sd.sm.valid['modelV2'] = fault != 'invalid_model'
  sd.sm.seen['carOutput'] = fault != 'unseen_output'
  sd.sm.alive['modelV2'] = fault != 'dead_model'
  ages = {service: 0 for service in SERVICES}
  if fault == 'old_control':
    ages['carControl'] = 30_000_001
  elif fault == 'old_output':
    ages['carOutput'] = 30_000_001
  elif fault == 'old_model':
    ages['modelV2'] = 30_000_001
  elif fault == 'old_state':
    ages['carState'] = 30_000_001
  elif fault == 'future_control':
    ages['carControl'] = -1
  assert not sample(sd, cs, 1_010_000_000,
    torque=float('nan') if fault == 'nan_torque' else 0.98 if fault == 'low_torque' else 1.0,
    curvature=float('nan') if fault == 'nan_curvature' else -0.0212,
    error=float('nan') if fault == 'nan_angle' else 8.0 if fault == 'small_error' else 30.0,
    ages=ages)
  assert sd.gv70_torque_saturation_since_ns == 0
  if fault in fields:
    setattr(cs, fields[fault][0], fault == 'can')
  sd.sm['carControl'].latActive = sd.conditional_car_state_valid = True
  sd.sm.valid['carOutput'] = sd.sm.valid['modelV2'] = True
  sd.sm.seen['carOutput'] = sd.sm.alive['modelV2'] = True
  assert not sample(sd, cs, 1_020_000_000)


@pytest.mark.parametrize('elapsed', (2_000_000_000, -10_000_000))
def test_gap_or_clock_reversal_cannot_complete_a_previous_interval(elapsed, monkeypatch):
  sd, cs = warning_state(monkeypatch)
  sample(sd, cs, 0)
  sample(sd, cs, 900_000_000)
  assert not sample(sd, cs, elapsed)


def test_gv70_torque_profile_enables_warning():
  with OpenpilotPrefix():
    cp = gv70_params()
    before = cp.to_dict()
    sd = SelfdriveD(CP=cp.as_reader())
    assert sd.gv70_torque_steering
    assert sd.gv70_vehicle_model is not None
    assert not sd.hkg_angle_steering
    assert cp.to_dict() == before


@pytest.mark.parametrize('vehicle', (CAR.GENESIS_GV70_1ST_GEN, CAR.KIA_EV6, CAR.HYUNDAI_IONIQ_5_PE))
def test_warning_does_not_enable_for_other_cars(vehicle):
  with OpenpilotPrefix():
    sd = SelfdriveD(CP=gv70_params(vehicle).as_reader())
    assert not sd.gv70_torque_steering
    assert sd.gv70_vehicle_model is None


@pytest.mark.parametrize('change', ('brand', 'angle', 'not_car'))
def test_warning_does_not_enable_for_other_configurations(change):
  with OpenpilotPrefix():
    cp = gv70_params().as_reader().as_builder()
    if change == 'brand':
      cp.brand = 'mock'
    elif change == 'angle':
      cp.steerControlType = car.CarParams.SteerControlType.angle
    else:
      cp.notCar = True
    assert not SelfdriveD(CP=cp.as_reader()).gv70_torque_steering


def test_existing_saturation_warning_is_not_duplicated(monkeypatch):
  sd, cs = warning_state(monkeypatch)
  for elapsed in range(0, 1_000_000_000, 10_000_000):
    sample(sd, cs, elapsed)
  sd.events.add(log.OnroadEvent.EventName.steerSaturated)
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=BASE + 1_000_000_000):
    sd._update_gv70_torque_saturation_warning(cs)
  assert sd.events.names.count(log.OnroadEvent.EventName.steerSaturated) == 1
