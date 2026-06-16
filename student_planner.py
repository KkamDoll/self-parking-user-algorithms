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
import heapq  # [B]
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
    """ 값을 [lo, hi] 범위로 자르기 """
    return max(lo, min(hi, x))


def _wrap_to_pi(a: float) -> float:  # [D]  --> -pi ~ +pi 로 정규화
    """ 각도를 정규화 (각도 오차 비교용) """
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


def gear_for_direction(direction: int) -> str:  # [D] --> 전진 후진 기어
    return "D" if direction >= 0 else "R"       # +면 "D", -면 "R"


def target_speed(dist_to_goal: float, dist_to_cusp: float, phase: str) -> float:  # [D]
    """거리 기반 목표 속도 (감속 프로파일)"""

    if phase == "STOP":
        return 0.0
    v = PARAMS["v_cruise"]
    v = min(v, PARAMS["k_cusp"] * dist_to_cusp + PARAMS["v_min"])  # 컵스 감속
    v = min(v, PARAMS["k_goal"] * dist_to_goal + PARAMS["v_min"])  # 골 크롤
    return _clamp(v, 0.0, PARAMS["v_cruise"])


def speed_to_pedal(v: float, v_tgt: float, direction: int):  # [D]
    """기어가 방향을 결정한다고 가정. ※ 시뮬 gear 처리 확인 후 부호 검증 필요.
        속도 오차 -> accel/brake (P제어)"""

    v_along = direction * v
    err = v_tgt - v_along
    if err > 0:
        return _clamp(PARAMS["kp_v"] * err, 0.0, 1.0), 0.0   # accel
    return 0.0, _clamp(-PARAMS["kp_v"] * err, 0.0, 1.0)       # brake


def reached_goal(state: Dict[str, Any], goal) -> bool:  # [D]
    """위치 + 방향 둘다 허용 오차 안이면 True"""

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


# ===========================================================================
# [B] 추가 영역 시작 — Hybrid A* 상수 / 탐색 노드
# ===========================================================================
_B_VEH_WIDTH = 2.0                               # [B] 차량 폭 (m)
_B_STEP = 0.5                                    # [B] 탐색 호 길이 (m)
_B_STEER_LIST = [                                # [B] 이산화된 조향각 목록
    -MAX_STEER, -MAX_STEER * 0.5, 0.0, MAX_STEER * 0.5, MAX_STEER
]
_B_XY_RES = 0.5                                  # [B] 위치 이산화 해상도 (m)
_B_YAW_RES = math.radians(5)                     # [B] 방향 이산화 해상도 (5도)
_B_N_YAW = int(round(2 * math.pi / _B_YAW_RES)) # [B] yaw bin 수 (72)
_B_REVERSE_COST = 2.0                            # [B] 후진 비용 배율
_B_CUSP_COST = 3.0                               # [B] 전/후진 전환 추가 비용
_B_MAX_ITER = 120000                             # [B] 최대 탐색 반복 횟수
_B_GOAL_XY = 0.6                                 # [B] 목표 위치 허용 오차 (m)
_B_GOAL_YAW = math.radians(10)                   # [B] 목표 방향 허용 오차 (rad)
_B_HEURISTIC_WEIGHT = 2.5                        # [B] 휴리스틱 가중치(>1=greedy, 탐색 가속)


class _HybridNode:  # [B] Hybrid A* 탐색 노드
    """Hybrid A* 탐색 노드 (최소 힙 지원)."""

    __slots__ = ('x', 'y', 'yaw', 'direction', 'g', 'f', 'parent')

    def __init__(self, x, y, yaw, direction, g, f, parent=None):
        self.x, self.y, self.yaw = x, y, yaw
        self.direction = direction  # +1(전진) / -1(후진)
        self.g, self.f = g, f
        self.parent = parent

    def __lt__(self, other):  # [B] 힙 비교
        return self.f < other.f

    def key(self):  # [B] 이산화된 상태 키 (방문 체크용)
        return (
            round(self.x / _B_XY_RES),
            round(self.y / _B_XY_RES),
            int(round(self.yaw / _B_YAW_RES)) % _B_N_YAW,
            0 if self.direction > 0 else 1,
        )
# ===========================================================================
# [B] 추가 영역 끝
# ===========================================================================


# [D] TODO [B] Hybrid A* — B 가 구현. (static_map, start_state, goal) -> 경로 리스트.
#     경로의 각 점은 .x .y .yaw .direction(+1/-1) 을 가져야 함. 미완성 시 [] 반환.
def hybrid_astar_plan(static_map, start_state, goal):  # [B] Hybrid A* 구현
    """비완전구동 차량용 Hybrid A* 경로 계획기.

    반환: [[x, y, yaw, direction], ...] 웨이포인트 목록 (실패 시 [])
    """
    if not static_map or not start_state or goal is None:
        return []

    # [B] 맵 정보 파싱
    extent = static_map.get("extent", [0, 10, 0, 10])
    xmin, _, ymin, _ = [float(v) for v in extent]
    cell = float(static_map.get("cellSize", _B_XY_RES))

    grid_data = static_map.get("grid", {})
    stat_layer = grid_data.get("stationary") or []
    park_layer = grid_data.get("parked") or []
    n_rows = len(stat_layer)
    n_cols = len(stat_layer[0]) if n_rows > 0 else 0

    # [B] 이진 장애물 flat 배열 (row-major, row=0 → ymin 방향)
    occ = bytearray(n_rows * n_cols)
    for r in range(n_rows):
        row_s = stat_layer[r]
        row_p = park_layer[r] if r < len(park_layer) else ()
        for c in range(n_cols):
            s = row_s[c] if c < len(row_s) else 0.0
            p = row_p[c] if c < len(row_p) else 0.0
            if s > 0.5 or p > 0.5:
                occ[r * n_cols + c] = 1

    def cell_occupied(x, y):
        if n_rows == 0 or n_cols == 0:
            return False  # 그리드 없으면 장애물 없음으로 처리
        c_i = int((x - xmin) / cell)
        r_i = int((y - ymin) / cell)
        if not (0 <= r_i < n_rows and 0 <= c_i < n_cols):
            return True  # 맵 밖 = 충돌로 처리
        return occ[r_i * n_cols + c_i] != 0

    hw = _B_VEH_WIDTH / 2

    def collision_free(x, y, yaw):
        """차량 footprint(종방향 5점 × 횡방향 3점)를 장애물 그리드에 체크."""
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        # 종방향: 후방(-P_LR) ~ 전방(P_LF), 0.5m 간격으로 5점
        for lon in (-P_LR, -P_LR * 0.5, 0.0, P_LF * 0.5, P_LF):
            bx = x + lon * cos_y
            by = y + lon * sin_y
            for lat in (-hw, 0.0, hw):  # 우측 / 중앙 / 좌측
                if cell_occupied(bx - lat * sin_y, by + lat * cos_y):
                    return False
        return True

    gx, gy, gyaw = goal

    def heuristic(x, y, yaw):
        dist = math.hypot(x - gx, y - gy)
        dyaw = abs(_wrap_to_pi(yaw - gyaw))
        # [B] greedy 가중치로 탐색을 목표 쪽으로 강하게 유도 (노드 수 급감)
        return _B_HEURISTIC_WEIGHT * (dist + dyaw * P_L * 0.3)  # 방향 오차를 호 길이로 근사

    # [B] 시작 노드 설정 및 탐색 초기화
    sx = float(start_state['x'])
    sy = float(start_state['y'])
    syaw = float(start_state['yaw'])

    start_node = _HybridNode(sx, sy, syaw, 1, 0.0, heuristic(sx, sy, syaw))
    open_heap = [start_node]
    best_g: Dict[tuple, float] = {}
    result = None

    for _ in range(_B_MAX_ITER):
        if not open_heap:
            break
        cur = heapq.heappop(open_heap)
        k = cur.key()
        if k in best_g and best_g[k] <= cur.g:
            continue
        best_g[k] = cur.g

        # [B] 목표 도달 확인
        if (math.hypot(cur.x - gx, cur.y - gy) < _B_GOAL_XY
                and abs(_wrap_to_pi(cur.yaw - gyaw)) < _B_GOAL_YAW):
            result = cur
            break

        # [B] 노드 확장: 2방향 × 5조향각 = 10 후계 노드
        for direction in (1, -1):
            d = direction * _B_STEP
            cusp = _B_CUSP_COST if direction != cur.direction else 0.0
            for steer in _B_STEER_LIST:
                nx = cur.x + d * math.cos(cur.yaw)
                ny = cur.y + d * math.sin(cur.yaw)
                nyaw = _wrap_to_pi(cur.yaw + d * math.tan(steer) / P_L)
                if not collision_free(nx, ny, nyaw):
                    continue
                ng = cur.g + _B_STEP * (_B_REVERSE_COST if direction < 0 else 1.0) + cusp
                child = _HybridNode(nx, ny, nyaw, direction, ng,
                                    ng + heuristic(nx, ny, nyaw), cur)
                ck = child.key()
                if ck not in best_g or best_g[ck] > ng:
                    heapq.heappush(open_heap, child)

    if result is None:
        print(f"[B] Hybrid A*: 경로 미발견 ({_B_MAX_ITER}회 탐색)")
        return []

    # [B] 경로 역추적 → [x, y, yaw, direction] 리스트 반환
    path = []
    node = result
    while node is not None:
        path.append([node.x, node.y, node.yaw, node.direction])
        node = node.parent
    path.reverse()
    print(f"[B] Hybrid A*: 경로 발견 ({len(path)} 웨이포인트)")
    return path


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

    # [B] 맵 분석 결과
    slots: Optional[List] = None       # [B] 전체 슬롯 목록
    free_slots: Optional[List] = None  # [B] 빈 슬롯 목록

    def __post_init__(self) -> None:
        if self.waypoints is None:
            self.waypoints = []
        # [D] D 상태 초기화 (원본 줄은 위에 그대로 유지)
        self.path = None
        self.idx = 0
        self.goal = None
        self.phase = "PLAN"
        # [B] B 상태 초기화
        self.slots = []
        self.free_slots = []

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

        # [B] 슬롯 파싱: 빈 슬롯 / 점유 슬롯 분류
        self.slots = list(map_payload.get("slots", []))
        occupied_idx = map_payload.get("occupied_idx", [])
        self.free_slots = [
            s for i, s in enumerate(self.slots)
            if not (i < len(occupied_idx) and occupied_idx[i])
        ]
        print(f"[B] slots parsed: {len(self.slots)} total, {len(self.free_slots)} free")

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다."""

        # TODO: A*, RRT*, Hybrid A* 등으로 self.waypoints를 채우세요.
        self.waypoints.clear()

    # [D] 추가: 주차 진입 로직 (오케스트레이션). 명령을 만들면 dict, 못 만들면 None.
    def _compute_control_D(self, obs: Dict[str, Any]):
        state = obs.get("state", {})
        if "x" not in state:
            return None  # state 없으면 원본 데모로 fallback

        # (a) 최초 1회: 목표 pose 정하고 Hybrid A* 호출 [connect to 민호(B)]
        if self.path is None:                   # 아직 경로가 없으면
            slot = obs.get("target_slot")       # 목표 주차 슬롯
            if slot is None:
                return None
            self.goal = select_goal_pose(slot, self.expected_orientation)   # 목표 위치 + 방향 계산
            self.path = hybrid_astar_plan(self.map_data, state, self.goal)  # [B] 호출해서 경로 받기
            self.idx = 0
            self.phase = "DRIVE"

        # (fallback) 경로가 비었으면(B 미완성) 안전 정지
        if not self.path:       # B가 미구현이라 [] 반환 -> 여기서 코드 ㄱ걸림
            return {"steer": 0.0, "accel": 0.0, "brake": 0.3, "gear": "D"}

        # (b) 정차 단계면 계속 브레이크
        if self.phase == "STOP" or reached_goal(state, self.goal):
            self.phase = "STOP"
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # (c) 경로 진행 + 현재 세그먼트 방향/기어
        self.idx = advance_index(self.path, state, self.idx)    # 가까워진 웨이포인트는 통과
        direction = _wp_dir(self.path[self.idx])                # 이 구간이 전진(+1) / 후진(-1)
        gear = gear_for_direction(direction)                    # D or R

        # (d) 속도 프로파일 (컵스/골 감속)
        d_goal = dist_to_goal(state, self.path[-1])             # 골까지 거리
        d_cusp = dist_to_next_cusp(self.path, self.idx)         # 다음 전후진 전환점(cusp)까지 거리
        v_tgt = target_speed(d_goal, d_cusp, self.phase)        # 목표 속도 = 둘 중 더 빡센 감속 적용

        # (e) 조향은 Pure Pursuit [C], 속도는 D 가 페달로
        steer = pure_pursuit_steer(state, self.path, self.idx, direction)  # [C] 조향각
        steer = _clamp(steer, -MAX_STEER, MAX_STEER)                       # 목표속도와 현재속도 차이로 페달 계산
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