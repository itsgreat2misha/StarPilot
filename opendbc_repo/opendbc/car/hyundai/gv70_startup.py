"""Exact first-generation GV70 Electrified captured-ADRV startup.

Experimental sequencing change (test branch only): unlike the shared
CapturedADRVStartup path, GV70 does NOT gate longitudinal takeover on a
pre-disable ADRV 0x51 capture. Instead it follows official StarPilot's
pre-7.0 shape — disable ADAS first, harvesting stock 0x51 frames that
flow while the disable transaction drains CAN, and send the ADRV
heartbeat at the first control frame regardless of whether a template
was harvested (empty 0x51 when none was, via create_adrv_messages'
existing template=None branch). This decouples "ADRV heartbeat on"
from "captured template ready" to shorten the ADAS-status gap at
startup that the cluster's self-check flags as a blind-spot warning.
It changes no panda safety code, no TX set, no message rate, no
AEB_SETTING, and no BSM/BCA status: the heartbeat is exactly the set
official StarPilot already sends. EV6 and all other cars are
unaffected (they still use CapturedADRVStartup.prepare).
"""
import time

from opendbc.car.hyundai.captured_adrv_startup import CapturedADRVStartup, first_generation
from opendbc.car.hyundai.ecu_continuity import ContinuousECUStartup
from opendbc.car.hyundai.ecu_startup import HyundaiECUStartup, Outcome
from opendbc.car.hyundai.gv70_template import GV70Template
from opendbc.car.hyundai.values import CAR


def eligible(cp):
  return cp.carFingerprint == CAR.GENESIS_GV70_ELECTRIFIED_1ST_GEN and first_generation(cp)


def required(cp):
  return eligible(cp) and cp.openpilotLongitudinalControl


class GV70Startup(CapturedADRVStartup):
  def __init__(self, cp, callbacks):
    super().__init__(cp, callbacks, label='GV70', template_type=GV70Template)
    self._harvest_floor_ns = 0
    self._harvest_previous = None

  def _harvest(self, packet):
    """Opportunistically validate one drained packet as a template source.

    Same validation as CapturedADRVStartup._capture (CRC via
    GV70Template.capture, <=150 ms fresh, two consecutive counters),
    but per packet and without a deadline or sleep loop: it only ever
    sees packets the disable transaction already drains.
    """
    if self.template is not None:
      return
    stamp = int(getattr(packet, 'log_mono_time_ns', 0))
    now = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    if not self._harvest_floor_ns < stamp <= now or now - stamp > 150_000_000:
      return
    for msg in packet:
      if msg.src != self.template_bus or msg.address != 0x51:
        continue
      try:
        candidate = self.template_type.capture(bytes(msg.dat))
      except ValueError:
        self._harvest_previous = None
        continue
      previous = self._harvest_previous
      if previous is not None and stamp > previous[0] and candidate.counter == (previous[1].counter + 1) % 256:
        self.template = candidate
        return
      self._harvest_previous = stamp, candidate

  def prepare(self, *, admission):
    # No pre-disable capture gate: pass the caller's admission straight
    # through to ContinuousECUStartup.prepare (disable, then keeper),
    # harvesting stock 0x51 frames while the disable drains CAN.
    # Deliberate trade-off vs Dom's fail-closed gate: if no template
    # is harvested by the time the disable succeeds, configure() below
    # installs an empty template and the ADRV heartbeat still flows at
    # the first control frame (official shape), instead of falling
    # back to stock with no ADRV at all.
    self._harvest_floor_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    self._harvest_previous = None
    recv, send = self.callbacks

    def harvesting_recv(*args, **kwargs):
      packets = recv(*args, **kwargs)
      if self.template is None:
        for packet in packets or []:
          self._harvest(packet)
          if self.template is not None:
            break
      return packets

    self.callbacks = (harvesting_recv, send)
    try:
      return ContinuousECUStartup.prepare(self, admission=admission)
    finally:
      self.callbacks = (recv, send)

  def configure(self, ci):
    if self.outcome is Outcome.SENT_UNCONFIRMED and self.template is None:
      # Empty-template continuation: HyundaiECUStartup.configure sets
      # ready for SENT_UNCONFIRMED; the carcontroller then sends an
      # empty ADRV 0x51 + the full ADRV heartbeat at the first frame.
      HyundaiECUStartup.configure(self, ci)
      ci.CC.adrv_template = None
    else:
      super().configure(ci)
    # This owner exists only for a requested exact-model takeover. The original
    # failed-disable controller avoided fighting stock cruise with Cancel.
    ci.CC.gv70_stock_fallback = self.outcome is not Outcome.SENT_UNCONFIRMED
