"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ----------------- 제어기 파라미터 -----------------
WHEELBASE_DEFAULT = 2.6
MAX_STEER_DEFAULT = math.radians(35.0)

CRUISE_SPEED = 1.5       # m/s, 주행 목표 속도
APPROACH_GAIN = 0.6      # 최종 목표 접근 시 거리->목표속도 변환 계수
CURVE_SLOWDOWN = 0.6     # 조향각이 클수록 목표 속도를 줄이는 비율

ACCEL_GAIN = 0.6         # 속도 오차 -> accel 비율 게인
BRAKE_GAIN = 0.8         # 속도 오차 -> brake 비율 게인

WAYPOINT_REACH_RADIUS = 1.0  # m, 이 거리 안으로 들어오면 다음 웨이포인트로 전환
GOAL_TOLERANCE = 0.3         # m, 최종 목표 도달 판정 거리
STOP_SPEED = 0.05            # m/s, 정지로 간주하는 속도

GEAR_SWITCH_SPEED = 0.3  # m/s, 이 속도 이하일 때만 기어 전환 허용
GEAR_HYSTERESIS = 0.2    # m, 기어 전환 판정용 히스테리시스


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)
    stationary = map_payload.get("grid", {}).get("stationary")
    if stationary:
        rows = len(stationary)
        cols = len(stationary[0]) if stationary else 0
        print("[algo] grid size  :", rows, "x", cols)


@dataclass
class PlannerSkeleton:
    """경로 계획/제어 로직을 담는 기본 스켈레톤 클래스입니다."""

    map_data: Optional[Dict[str, Any]] = None
    map_extent: Optional[Tuple[float, float, float, float]] = None
    cell_size: float = 0.5
    stationary_grid: Optional[List[List[float]]] = None
    waypoints: List[Tuple[float, float]] = None

    # --- 제어기 내부 상태 ---
    target_idx: int = 0
    gear: str = "D"

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터에서 전송한 정적 맵 데이터를 보관합니다."""

        self.map_data = map_payload
        self.map_extent = tuple(
            map(float, map_payload.get("extent", (0.0, 0.0, 0.0, 0.0)))
        )
        self.cell_size = float(map_payload.get("cellSize", 0.5))
        self.stationary_grid = map_payload.get("grid", {}).get("stationary")
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()
        self.target_idx = 0
        self.gear = "D"

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""

        # TODO: A*, RRT*, Hybrid A* 등으로 self.waypoints를 채우세요.
        self.waypoints.clear()

    def _advance_target(self, x: float, y: float) -> None:
        """현재 위치에서 도달한 웨이포인트를 건너뛰고 다음 목표를 가리킵니다."""

        self.target_idx = min(self.target_idx, len(self.waypoints) - 1)
        while self.target_idx < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.target_idx]
            if math.hypot(wx - x, wy - y) <= WAYPOINT_REACH_RADIUS:
                self.target_idx += 1
            else:
                break

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Pure Pursuit 기반으로 경로를 따라가는 조향/가감속 명령을 산출합니다."""

        state = obs.get("state", {})
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        yaw = float(state.get("yaw", 0.0))
        v = float(state.get("v", 0.0))

        limits = obs.get("limits", {})
        wheelbase = float(limits.get("L", WHEELBASE_DEFAULT))
        max_steer = float(limits.get("maxSteer", MAX_STEER_DEFAULT))

        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": self.gear}

        if not self.waypoints:
            cmd["brake"] = 0.3 if abs(v) > STOP_SPEED else 0.0
            return cmd

        self._advance_target(x, y)
        target_x, target_y = self.waypoints[self.target_idx]
        is_final = self.target_idx == len(self.waypoints) - 1

        dx, dy = target_x - x, target_y - y
        distance = math.hypot(dx, dy)

        # 차량 좌표계로 변환 (전방 +x, 좌측 +y)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        # 목표 지점이 후방이면 후진. 정지 상태(속도 충분히 낮을 때)에서만 전환 허용.
        desired_gear = "D" if local_x >= -GEAR_HYSTERESIS else "R"
        if desired_gear != self.gear and abs(v) > GEAR_SWITCH_SPEED:
            cmd["gear"] = self.gear
            cmd["brake"] = 1.0
            return cmd
        self.gear = desired_gear
        cmd["gear"] = self.gear

        # Pure Pursuit 조향각: steer = atan2(2*L*sin(alpha), Ld)
        alpha = math.atan2(local_y, local_x)
        steer = math.atan2(2.0 * wheelbase * math.sin(alpha), max(distance, 1e-3))
        cmd["steer"] = clamp(steer, -max_steer, max_steer)

        if is_final and distance < GOAL_TOLERANCE:
            cmd["brake"] = 1.0 if abs(v) > STOP_SPEED else 0.0
            return cmd

        # 종방향 제어: 목표 속도를 정하고 P 제어로 accel/brake 산출
        target_speed = CRUISE_SPEED * (1.0 - CURVE_SLOWDOWN * abs(cmd["steer"]) / max_steer)
        if is_final:
            target_speed = min(target_speed, distance * APPROACH_GAIN)

        speed_error = target_speed - abs(v)
        if speed_error > 0.0:
            cmd["accel"] = clamp(ACCEL_GAIN * speed_error, 0.0, 1.0)
        else:
            cmd["brake"] = clamp(BRAKE_GAIN * -speed_error, 0.0, 1.0)

        return cmd


# 전역 planner 인스턴스 (통신 모듈이 이 객체를 사용합니다.)
planner = PlannerSkeleton()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    """통신 모듈에서 맵 패킷을 받을 때 호출됩니다."""

    planner.set_map(map_payload)


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """통신 모듈에서 매 스텝 호출하여 명령을 생성합니다."""

    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}

