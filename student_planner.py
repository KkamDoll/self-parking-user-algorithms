"""학생 자율주차 알고리즘 스켈레톤 모듈.

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다. 학생은 아래 `PlannerSkeleton` 클래스나 `planner_step`
함수를 원하는 로직으로 교체/확장하면 됩니다.

------------------------------------------------------------------------
[D 파트 추가본] 원본 코드는 한 줄도 삭제하지 않았고, D(주차 진입 로직)
관련 내용만 "추가"했습니다. 추가된 부분은 모두  # [D]  주석으로 표시.
  - import math
  - 차량 상수 / PARAMS / D 헬퍼 함수들
  - PlannerSkeleton 에 D 상태 필드 + __post_init__/set_map 에 D 초기화 추가
  - compute_control 맨 위에 D 훅 추가 (원본 데모 제어는 아래에 그대로 보존)
B(Hybrid A*) / C(Pure Pursuit) 자리는 TODO 로 비워둠 -> 비어 있어도 실행됨.
------------------------------------------------------------------------
"""

import math  # [D] 추가
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# [D] 추가 영역 시작 — 차량 상수 / 튜닝 파라미터 / 헬퍼 함수
# ===========================================================================
P_L = 2.6        # [D] wheelbase
P_LF = 1.6       # [D] 후축->앞 오버행
P_LR = 1.4       # [D] 후축->뒤 오버행
MAX_STEER = 0.6109  # [D] rad(35deg). obs["limits"]["maxSteer"] 로 갱신 권장

# [D] 튜닝 파라미터 — 여기만 만지면서 점수 최적화 (리플레이 분석 + grid search)
PARAMS = {
    "v_cruise": 1.8,   # 일반 주행 목표 속도(m/s)
    "v_min": 0.4,      # 크롤 최저 속도
    "k_goal": 0.6,     # 골까지 거리 1m당 허용 속도 증가량
    "k_cusp": 0.8,     # 전/후진 전환점(컵스)까지 거리 1m당 허용 속도 증가량
    "kp_v": 1.2,       # 속도오차 -> 페달 P 게인
    "pos_tol": 0.25,   # ★ 골 위치 오차(m) — IoU 직결
    "yaw_tol": 0.15,   # ★ 골 헤딩 오차(rad, ~8.6deg)
    "stop_speed": 0.05,  # 이 속도 이하면 정차 완료로 간주
    "reach_radius": 0.6,  # 웨이포인트 통과 인정 반경(m)
}


def _clamp(x: float, lo: float, hi: float) -> float:  # [D]
    return max(lo, min(hi, x))


def _wrap_to_pi(a: float) -> float:  # [D]
    return (a + math.pi) % (2 * math.pi) - math.pi


def select_goal_pose(slot, expected_orientation: str):  # [D]
    """슬롯 [xmin,xmax,ymin,ymax] 과 요구 방향으로 목표 pose 계산.

    시뮬 determine_parking_orientation() 기준 (세로 슬롯이면 sin(yaw) 부호):
      front_in -> yaw=+90deg, rear_in -> yaw=-90deg
    """
    xmin, xmax, ymin, ymax = slot
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    width = xmax - xmin
    height = ymax - ymin
    if height >= width:  # 세로 슬롯(기본형)
        goal_yaw = math.pi / 2 if expected_orientation == "front_in" else -math.pi / 2
    else:                # 가로 슬롯
        goal_yaw = 0.0 if expected_orientation == "front_in" else math.pi
    # 차체 중심(후축에서 0.1m 앞)을 슬롯 중심에 맞춰 IoU 최대화
    body_offset = 0.5 * (P_LF - P_LR)  # = 0.1 m
    gx = cx - body_offset * math.cos(goal_yaw)
    gy = cy - body_offset * math.sin(goal_yaw)
    return (gx, gy, goal_yaw)


def gear_for_direction(direction: int) -> str:  # [D]
    return "D" if direction >= 0 else "R"


def target_speed(dist_to_goal: float, dist_to_cusp: float, phase: str) -> float:  # [D]
    if phase == "STOP":
        return 0.0
    v = PARAMS["v_cruise"]
    v = min(v, PARAMS["k_cusp"] * dist_to_cusp + PARAMS["v_min"])  # 컵스 감속
    v = min(v, PARAMS["k_goal"] * dist_to_goal + PARAMS["v_min"])  # 골 크롤
    return _clamp(v, 0.0, PARAMS["v_cruise"])


def speed_to_pedal(v: float, v_tgt: float, direction: int):  # [D]
    """기어가 방향을 결정한다고 가정. ※ 시뮬 gear 처리 확인 후 부호 검증 필요."""
    v_along = direction * v
    err = v_tgt - v_along
    if err > 0:
        return _clamp(PARAMS["kp_v"] * err, 0.0, 1.0), 0.0   # accel
    return 0.0, _clamp(-PARAMS["kp_v"] * err, 0.0, 1.0)       # brake


def reached_goal(state: Dict[str, Any], goal) -> bool:  # [D]
    dist = math.hypot(state["x"] - goal[0], state["y"] - goal[1])
    yaw_err = abs(_wrap_to_pi(state["yaw"] - goal[2]))
    return dist < PARAMS["pos_tol"] and yaw_err < PARAMS["yaw_tol"]


def _wp_xy(wp):  # [D] 웨이포인트가 객체(.x/.y)든 튜플이든 좌표 추출
    if hasattr(wp, "x"):
        return wp.x, wp.y
    return wp[0], wp[1]


def _wp_dir(wp) -> int:  # [D] 방향(+1 전진/-1 후진) 추출
    if hasattr(wp, "direction"):
        return wp.direction
    if isinstance(wp, (list, tuple)) and len(wp) >= 4:
        return int(wp[3])
    return 1


def advance_index(path, state, idx: int) -> int:  # [D]
    while idx < len(path) - 1:
        wx, wy = _wp_xy(path[idx])
        if math.hypot(state["x"] - wx, state["y"] - wy) < PARAMS["reach_radius"]:
            idx += 1
        else:
            break
    return idx


def dist_to_next_cusp(path, idx: int) -> float:  # [D]
    if idx >= len(path) - 1:
        return 0.0
    total = 0.0
    cur_dir = _wp_dir(path[idx])
    for i in range(idx, len(path) - 1):
        if _wp_dir(path[i + 1]) != cur_dir:
            break
        x1, y1 = _wp_xy(path[i])
        x2, y2 = _wp_xy(path[i + 1])
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def dist_to_goal(state, goal_wp) -> float:  # [D]
    gx, gy = _wp_xy(goal_wp)
    return math.hypot(state["x"] - gx, state["y"] - gy)


# [D] TODO [B] Hybrid A* — B 가 구현. (static_map, start_state, goal) -> 경로 리스트.
#     경로의 각 점은 .x .y .yaw .direction(+1/-1) 을 가져야 함. 미완성 시 [] 반환.
def hybrid_astar_plan(static_map, start_state, goal):  # [D] placeholder
    return []


# [D] TODO [C] Pure Pursuit — C 가 구현. (state, path, idx, direction) -> steer(rad).
def pure_pursuit_steer(state, path, idx, direction) -> float:  # [D] placeholder
    return 0.0
# ===========================================================================
# [D] 추가 영역 끝
# ===========================================================================


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

    # [D] 추가 상태 필드 (모두 기본값이 있어 dataclass 정상 동작)
    expected_orientation: str = "front_in"   # [D] set_map 에서 갱신
    path: Optional[list] = None              # [D] Hybrid A* 결과
    idx: int = 0                             # [D] 현재 웨이포인트 인덱스
    goal: Optional[Tuple[float, float, float]] = None  # [D] 목표 pose
    phase: str = "PLAN"                       # [D] PLAN -> DRIVE -> STOP

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []
        # [D] D 상태 초기화 (원본 줄은 위에 그대로 유지)
        self.path = None
        self.idx = 0
        self.goal = None
        self.phase = "PLAN"

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

        # [D] stage 별 주차 방향 저장 + 새 맵마다 경로/상태 리셋
        self.expected_orientation = map_payload.get("expected_orientation", "front_in")
        self.path = None
        self.idx = 0
        self.goal = None
        self.phase = "PLAN"

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""

        # TODO: A*, RRT*, Hybrid A* 등으로 self.waypoints를 채우세요.
        self.waypoints.clear()

    # [D] 추가: 주차 진입 로직 (오케스트레이션). 명령을 만들면 dict, 못 만들면 None.
    def _compute_control_D(self, obs: Dict[str, Any]):
        state = obs.get("state", {})
        if "x" not in state:
            return None  # state 없으면 원본 데모로 fallback

        # (a) 최초 1회: 목표 pose 정하고 Hybrid A* 호출 [B]
        if self.path is None:
            slot = obs.get("target_slot")
            if slot is None:
                return None
            self.goal = select_goal_pose(slot, self.expected_orientation)
            self.path = hybrid_astar_plan(self.map_data, state, self.goal)  # [B]
            self.idx = 0
            self.phase = "DRIVE"

        # (fallback) 경로가 비었으면(B 미완성) 안전 정지
        if not self.path:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.3, "gear": "D"}

        # (b) 정차 단계면 계속 브레이크
        if self.phase == "STOP" or reached_goal(state, self.goal):
            self.phase = "STOP"
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # (c) 경로 진행 + 현재 세그먼트 방향/기어
        self.idx = advance_index(self.path, state, self.idx)
        direction = _wp_dir(self.path[self.idx])
        gear = gear_for_direction(direction)

        # (d) 속도 프로파일 (컵스/골 감속)
        d_goal = dist_to_goal(state, self.path[-1])
        d_cusp = dist_to_next_cusp(self.path, self.idx)
        v_tgt = target_speed(d_goal, d_cusp, self.phase)

        # (e) 조향은 Pure Pursuit [C], 속도는 D 가 페달로
        steer = pure_pursuit_steer(state, self.path, self.idx, direction)  # [C]
        steer = _clamp(steer, -MAX_STEER, MAX_STEER)
        accel, brake = speed_to_pedal(state.get("v", 0.0), v_tgt, direction)
        return {"steer": steer, "accel": accel, "brake": brake, "gear": gear}

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""

        # [D] 먼저 D 로직 시도. 명령을 만들면 그걸 사용.
        d_cmd = self._compute_control_D(obs)
        if d_cmd is not None:
            return d_cmd

        # ===== 아래는 원본 데모 제어 (한 줄도 삭제하지 않고 보존) =====
        # 예시: 기본 데모 제어. 학생은 원하는 알고리즘으로 대체하면 됩니다.
        t = float(obs.get("t", 0.0))
        v = float(obs.get("state", {}).get("v", 0.0))

        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}

        if t < 2.0:
            cmd["accel"] = 0.6
        elif t < 3.0:
            cmd["brake"] = 0.3
        else:
            cmd["steer"] = 0.07
            if v < 1.0:
                cmd["accel"] = 0.2

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