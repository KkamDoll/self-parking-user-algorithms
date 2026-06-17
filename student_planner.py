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
B(Hybrid A*) 자리는 TODO 로 비워둠. C(Pure Pursuit)는 아래에 구현됨.
------------------------------------------------------------------------
"""

import math  # [D] 추가
import heapq  # [B]
import threading  # [E] 비동기 경로계획 (IPC recv 타임아웃 0.15s 회피)
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
    "v_cruise": 1.2,   # 일반 주행 목표 속도(m/s) — target_speed 기본값(곡률 미지정 시)
    # [G] 곡률 적응 속도: 직선은 빠르게(가속↑), 곡선/주차기동은 느리게(추종오차↓ → 충돌↓)
    "v_straight": 3.0,   # 직선(저곡률) 목표 속도(m/s). 직선 가속 강화(곡선은 자동 감속)
    "v_curve_min": 0.7,  # 급곡선(최소회전반경급) 목표 속도(m/s)
    # [G] 추종오차 기반 감속: 경로에서 벗어날수록 느리게 -> 좁은 곳 충돌 전 회복 시간 확보
    "cte_slow_gain": 1.6,   # 추종오차 1m 당 속도 상한 감소 비율
    "cte_slow_floor": 0.3,  # 최소 속도 비율(이 이하론 안 줄임)
    "v_min": 0.4,      # 크롤 최저 속도
    "k_goal": 0.6,     # 골까지 거리 1m당 허용 속도 증가량
    "k_cusp": 0.8,     # 전/후진 전환점(컵스)까지 거리 1m당 허용 속도 증가량
    "kp_v": 1.2,       # 속도오차 -> 페달 P 게인
    "pos_tol": 0.25,   # ★ 골 위치 오차(m) — IoU 직결
    "yaw_tol": 0.15,   # ★ 골 헤딩 오차(rad, ~8.6deg)
    "stop_speed": 0.05,  # 이 속도 이하면 정차 완료로 간주
    "reach_radius": 0.6,  # 웨이포인트 통과 인정 반경(m)
    # [E] Pure Pursuit 동적 look-ahead (속도가 빠를수록 멀리 봄). 추종 안정성 ↔ 코너 추종.
    "ld_min": 1.2,     # 최소 전방주시거리(m)
    "ld_max": 3.0,     # 최대 전방주시거리(m)
    "ld_k": 0.9,       # 속도 1m/s 당 전방주시거리 증가량
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


def target_speed(dist_to_goal: float, dist_to_cusp: float, phase: str,
                 v_max: float = None) -> float:  # [D][G]
    """거리 기반 목표 속도 (감속 프로파일). v_max=곡률 적응 상한(없으면 v_cruise)."""

    if phase == "STOP":
        return 0.0
    if v_max is None:
        v_max = PARAMS["v_cruise"]
    v = v_max
    v = min(v, PARAMS["k_cusp"] * dist_to_cusp + PARAMS["v_min"])  # 컵스 감속
    v = min(v, PARAMS["k_goal"] * dist_to_goal + PARAMS["v_min"])  # 골 크롤
    return _clamp(v, 0.0, v_max)


def curvature_speed(path, idx: int, seg_end: int) -> float:  # [G]
    """현재 위치 앞쪽(~2.5m) 경로의 최대 곡률로 목표 속도 상한을 정한다.
    직선(곡률 0) -> v_straight, 최소회전반경급 곡률 -> v_curve_min 으로 선형 보간.
    """
    look_m, acc, max_k = 2.5, 0.0, 0.0
    i = idx
    while i + 2 <= seg_end and acc < look_m:
        x0, y0 = _wp_xy(path[i]); x1, y1 = _wp_xy(path[i + 1]); x2, y2 = _wp_xy(path[i + 2])
        ds = math.hypot(x2 - x1, y2 - y1)
        if ds > 1e-6:
            dth = abs(_wrap_to_pi(math.atan2(y2 - y1, x2 - x1) - math.atan2(y1 - y0, x1 - x0)))
            max_k = max(max_k, dth / ds)
        acc += math.hypot(x1 - x0, y1 - y0)
        i += 1
    k_ref = math.tan(MAX_STEER) / P_L                  # 최소회전반경 곡률(~0.27)
    frac = _clamp(max_k / k_ref, 0.0, 1.0) if k_ref > 1e-6 else 0.0
    v_str, v_crv = PARAMS["v_straight"], PARAMS["v_curve_min"]
    return v_str - (v_str - v_crv) * frac


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


def segment_end(path, idx: int) -> int:  # [E] 현재 방향 세그먼트의 마지막 인덱스(다음 cusp 직전)
    d = _wp_dir(path[idx])
    j = idx
    while j < len(path) - 1 and _wp_dir(path[j + 1]) == d:
        j += 1
    return j


def segment_start(path, idx: int) -> int:  # [F] 현재 방향 세그먼트의 시작 인덱스(이전 cusp 직후)
    d = _wp_dir(path[idx])
    j = idx
    while j > 0 and _wp_dir(path[j - 1]) == d:
        j -= 1
    return j


def lookahead_target(path, state, idx: int, seg_end: int, ld: float) -> int:  # [E]
    """현재 세그먼트 [idx..seg_end] 안에서 차로부터 ld 만큼 앞선 look-ahead 점.

    cusp(세그먼트 끝)를 넘어가지 않도록 seg_end 로 클램프한다 — look-ahead 가 반대 방향
    세그먼트로 튀어 차가 경로를 이탈하던 문제를 막는다.
    """
    tgt = idx
    while tgt < seg_end:
        wx, wy = _wp_xy(path[tgt])
        if math.hypot(state["x"] - wx, state["y"] - wy) >= ld:
            break
        tgt += 1
    return tgt


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
_B_VEH_WIDTH = 1.6                               # [B] 차량 폭 (m). 시뮬 Params.W 와 동일
_B_INFLATE = 0.25                                # [E] footprint 안전여유(추종오차 흡수, m).
                                                 #     슬롯 폭 여유(~0.3m)보다 작아야 골 진입 가능
_B_STEP = 0.4                                    # [E] 탐색 호 길이 (m). 작을수록 매끄러운 경로
_B_STEER_LIST = [                                # [B] 이산화된 조향각 목록
    -MAX_STEER, -MAX_STEER * 0.5, 0.0, MAX_STEER * 0.5, MAX_STEER
]
_B_XY_RES = 0.3                                  # [E] 위치 이산화 해상도 (m). 좁은 통로 탐색용
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
def hybrid_astar_plan(static_map, start_state, goal, block_lines=True):  # [B] Hybrid A* 구현
    """비완전구동 차량용 Hybrid A* 경로 계획기.

    block_lines=True 면 painted line(타깃 슬롯 경계 제외)도 장애물로 취급한다.
    반환: [[x, y, yaw, direction], ...] 웨이포인트 목록 (실패 시 [])
    """
    if not static_map or not start_state or goal is None:
        return []

    # [B] 맵 정보 파싱
    extent = static_map.get("extent", [0, 10, 0, 10])
    xmin, xmax_w, ymin, ymax_w = [float(v) for v in extent]
    cell = float(static_map.get("cellSize", _B_XY_RES))

    grid_data = static_map.get("grid", {})
    stat_layer = grid_data.get("stationary") or []
    n_rows = len(stat_layer)
    n_cols = len(stat_layer[0]) if n_rows > 0 else 0

    # [B] 정지물(벽) 점유 격자 (row-major, 시뮬 규약상 row=0 → ymax(맵 위쪽)).
    #     [E] 시뮬 detect_stationary_collision 과 동일하게 "stationary 레이어만" 사용한다.
    #         (주차차량은 점유 슬롯 사각형으로, parked 격자는 시각용이라 제외)
    occ = bytearray(n_rows * n_cols)
    for r in range(n_rows):
        row_s = stat_layer[r]
        for c in range(n_cols):
            if (row_s[c] if c < len(row_s) else 0.0) > 0.5:
                occ[r * n_cols + c] = 1

    # [E] 장애물 사각형 목록(래스터화 X, 정밀 SAT 용). 시뮬은 차체 폴리곤 vs 사각형 SAT 로
    #     충돌을 판정하므로(poly_intersects_rect), 0.5m 격자 래스터화의 반올림 오차 없이
    #     동일한 기하 검사를 하려면 사각형을 그대로 들고 검사해야 한다.
    slots = static_map.get("slots") or []
    occ_idx = static_map.get("occupied_idx") or []
    slot_rects = [tuple(float(v) for v in rect)
                  for i, rect in enumerate(slots)
                  if i < len(occ_idx) and occ_idx[i]]  # 점유 슬롯(주차차량)

    line_rects_obs = []  # painted line (타깃 슬롯 경계선 제외)
    if block_lines:
        tgt = None  # 타깃 슬롯 = goal(body center)을 포함하는 슬롯
        for rect in slots:
            rx0, rx1, ry0, ry1 = (float(v) for v in rect)
            if rx0 <= goal[0] <= rx1 and ry0 <= goal[1] <= ry1:
                tgt = (rx0, rx1, ry0, ry1)
                break
        L_HW = 0.25  # compute_line_rects 의 half-width 와 동일
        for seg in (static_map.get("lines") or []):
            x1, y1, x2, y2 = (float(v) for v in seg)
            if abs(x1 - x2) < 1e-6:      # 수직 선분
                a, b, c, d = min(x1, x2) - L_HW, max(x1, x2) + L_HW, min(y1, y2), max(y1, y2)
            elif abs(y1 - y2) < 1e-6:    # 수평 선분
                a, b, c, d = min(x1, x2), max(x1, x2), min(y1, y2) - L_HW, max(y1, y2) + L_HW
            else:
                a, b, c, d = (min(x1, x2) - L_HW, max(x1, x2) + L_HW,
                              min(y1, y2) - L_HW, max(y1, y2) + L_HW)
            if tgt is not None and not (b < tgt[0] or a > tgt[1] or d < tgt[2] or c > tgt[3]):
                continue  # 타깃 슬롯과 겹치는 라인은 진입 통로이므로 막지 않는다
            line_rects_obs.append((a, b, c, d))

    # [E] 부풀린 차량 OBB 의 로컬 꼭짓점 (안전여유 _B_INFLATE 포함, 1회 선계산)
    _fL, _bL = P_LF + _B_INFLATE, P_LR + _B_INFLATE
    _wH = _B_VEH_WIDTH / 2 + _B_INFLATE
    _car_local = ((_fL, _wH), (_fL, -_wH), (-_bL, -_wH), (-_bL, _wH))

    def _obb_hits_rect(corners, ax_cos, ax_sin, rx0, rx1, ry0, ry1):
        """차량 OBB(corners) vs 축정렬 사각형 SAT. 분리축 4개(사각형 2 + 차량 2)."""
        cx = [p[0] for p in corners]; cy = [p[1] for p in corners]
        if max(cx) < rx0 or min(cx) > rx1 or max(cy) < ry0 or min(cy) > ry1:
            return False                                  # 사각형 축(AA bbox)으로 분리
        rect_pts = ((rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1))
        for ux, uy in ((ax_cos, ax_sin), (-ax_sin, ax_cos)):  # 차량 두 축
            cmn = cmx = corners[0][0] * ux + corners[0][1] * uy
            for px, py in corners[1:]:
                d = px * ux + py * uy
                cmn = d if d < cmn else cmn
                cmx = d if d > cmx else cmx
            rmn = rmx = rect_pts[0][0] * ux + rect_pts[0][1] * uy
            for px, py in rect_pts[1:]:
                d = px * ux + py * uy
                rmn = d if d < rmn else rmn
                rmx = d if d > rmx else rmx
            if cmx < rmn or rmx < cmn:
                return False
        return True

    def collision_free(x, y, yaw):
        """차량 OBB(부풀림) vs 정지셀/점유슬롯/라인 사각형 정밀 SAT (시뮬과 동일 기하)."""
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        corners = [(x + lx * cos_y - ly * sin_y, y + lx * sin_y + ly * cos_y)
                   for lx, ly in _car_local]
        cxs = [p[0] for p in corners]; cys = [p[1] for p in corners]
        bx0, bx1 = min(cxs), max(cxs)
        by0, by1 = min(cys), max(cys)
        if bx0 < xmin or bx1 > xmax_w or by0 < ymin or by1 > ymax_w:
            return False                                  # 맵 밖 = 충돌
        # 1) 정지물 벽: 차량 bbox 안의 점유 셀만 정밀 SAT (시뮬 detect_stationary_collision)
        if n_rows and n_cols:
            c0 = max(0, int((bx0 - xmin) / cell)); c1 = min(n_cols - 1, int((bx1 - xmin) / cell))
            r0 = max(0, int((ymax_w - by1) / cell)); r1 = min(n_rows - 1, int((ymax_w - by0) / cell))
            for r in range(r0, r1 + 1):
                cell_ymax = ymax_w - r * cell; cell_ymin = cell_ymax - cell
                base = r * n_cols
                for c in range(c0, c1 + 1):
                    if occ[base + c] and _obb_hits_rect(
                            corners, cos_y, sin_y,
                            xmin + c * cell, xmin + c * cell + cell, cell_ymin, cell_ymax):
                        return False
        # 2) 점유 슬롯(주차차량)
        for sx0, sx1, sy0, sy1 in slot_rects:
            if bx1 < sx0 or bx0 > sx1 or by1 < sy0 or by0 > sy1:
                continue
            if _obb_hits_rect(corners, cos_y, sin_y, sx0, sx1, sy0, sy1):
                return False
        # 3) painted line
        for lx0, lx1, ly0, ly1 in line_rects_obs:
            if bx1 < lx0 or bx0 > lx1 or by1 < ly0 or by0 > ly1:
                continue
            if _obb_hits_rect(corners, cos_y, sin_y, lx0, lx1, ly0, ly1):
                return False
        return True

    gx, gy, gyaw = goal

    # [F] Dijkstra 거리장 휴리스틱은 실측상 회귀(라인 우회 최단경로가 장애물을 끼고 돌아 추종
    #     불가 -> 18/18 충돌)라 사용하지 않는다. Euclidean greedy 가 추종 가능한 경로를 더 잘 냄.
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


# [C] Pure Pursuit — 현재 위치에서 목표 웨이포인트(path[idx])까지의 조향각(rad) 반환.
#     시그니처는 D 컨트랙트 (state, path, idx, direction) 를 그대로 따른다.
#     전진(direction>=0) / 후진(direction<0) 모두 처리하며, 속도/기어/정차 판단은
#     D 오케스트레이션(_compute_control_D)이 담당하고 여기서는 조향만 계산한다.
#     반환값 클램프(-MAX_STEER~+MAX_STEER)는 호출부(_compute_control_D)에서 수행.
def pure_pursuit_steer(state, path, idx, direction) -> float:  # [C] 구현
    '''
    목표점(look-ahead point) 를 향해 가는 방법
    
    Pure Pursuit 조향각 계산.

    공식: delta = atan2(2 * L * sin(alpha), L_d)
    - delta : 조향각        - L   : 차량 축간거리(Wheelbase)
    - L_d   : 전방주시거리   - alpha : 헤딩과 목표점 사이 사잇각

    기하 유도:
    1) 곡률 kappa = 1/R = 2*sin(alpha)/L_d
    2) 자전거 모델 tan(delta) = L/R = L*kappa
    3) delta = atan(L*kappa)
    '''

    x = float(state.get("x", 0.0))           # 현재 차 위치 
    y = float(state.get("y", 0.0))
    yaw = float(state.get("yaw", 0.0))       # 현재 차 방향

    tx, ty = _wp_xy(path[idx])               # [D] 헬퍼 재사용 (객체/튜플 모두 지원)
                                             # 따라갈 목표의 웨이포인트

    dx, dy = tx - x, ty - y                  # 목표까지의 벡터
    dist = math.hypot(dx, dy)                # 목표까지의 거리 = L_d

    """차량 좌표계로 변환 (전방 +x, 좌측 +y) + 목표점까지의 방향과 거리를 구함"""
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)    
    local_x = dx * cos_yaw + dy * sin_yaw    # 차 기준 전방 성분
    local_y = -dx * sin_yaw + dy * cos_yaw   # 차 기준 좌측 성분

    if direction < 0:       
    # 후진: 목표를 후축 뒤쪽에서 바라보고 계산한 뒤 조향을 반전
        alpha = math.atan2(-local_y, -local_x)                              # 각도 오차 α
        return -math.atan2(2.0 * P_L * math.sin(alpha), max(dist, 1e-3))    # 조향각 δ

    # 전진: Pure Pursuit  steer = atan2(2*L*sin(alpha), Ld)
    alpha = math.atan2(local_y, local_x)
    return math.atan2(2.0 * P_L * math.sin(alpha), max(dist, 1e-3))


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
        # [E] 비동기 경로계획 상태 (백그라운드 스레드에서 계획 -> 매 스텝 즉시 응답)
        self._planning = False      # 계획 스레드가 실행 중인가
        self._plan_ready = False    # 결과가 준비됐는가
        self._plan_result = None    # 계획 결과 (경로 리스트 또는 [])
        self._plan_gen = 0          # 맵 교체 시 이전 스레드 결과를 무효화하는 토큰

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
        # [E] 새 맵/스테이지마다 비동기 계획 상태 리셋 + 진행 중 스레드 결과 무효화
        self._planning = False
        self._plan_ready = False
        self._plan_result = None
        self._plan_gen += 1

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

        # (a) 최초 1회: 목표 pose 정하고 Hybrid A* 를 "백그라운드 스레드"에서 호출.
        #     [E] Hybrid A* 는 수십~수천 ms 걸릴 수 있는데 시뮬레이터 recv 타임아웃은 0.15s 라,
        #     동기로 돌리면 첫 스텝에서 연결이 끊긴다. 그래서 계획은 스레드에 맡기고,
        #     계획이 끝날 때까지 매 스텝 즉시 "정지 유지" 명령을 반환한다.
        if self.path is None:                   # 아직 경로가 없으면
            slot = obs.get("target_slot")       # 목표 주차 슬롯
            if slot is None:
                return None

            if not self._planning:              # 계획 스레드 시작 (최초 1회)
                self.goal = select_goal_pose(slot, self.expected_orientation)  # 목표 pose
                start_state = {"x": state["x"], "y": state["y"],
                               "yaw": state.get("yaw", 0.0)}  # 스레드에 넘길 스냅샷(복사)
                self._planning = True
                self._plan_ready = False
                self._plan_result = None
                self._plan_gen += 1
                gen = self._plan_gen

                def _worker(gen=gen, map_data=self.map_data,
                            start_state=start_state, goal=self.goal):
                    try:
                        # [E] 라인까지 피하는 경로를 먼저 시도(깨끗한 주차).
                        #     못 찾으면 라인 무시 경로로 폴백(최소한 목표로 시도).
                        p = hybrid_astar_plan(map_data, start_state, goal, block_lines=True)
                        if not p:
                            print("[E] 라인 회피 경로 미발견 -> 라인 무시 폴백")
                            p = hybrid_astar_plan(map_data, start_state, goal, block_lines=False)
                    except Exception as exc:
                        print(f"[E] planning thread error: {exc}")
                        p = []
                    if gen == self._plan_gen:   # 최신 요청일 때만 반영 (맵 교체 race 방지)
                        self._plan_result = p
                        self._plan_ready = True

                threading.Thread(target=_worker, daemon=True).start()

            if not self._plan_ready:            # 아직 계획 중 -> 제자리 정지 유지
                return {"steer": 0.0, "accel": 0.0, "brake": 0.3, "gear": "D"}

            # 계획 완료 -> 결과 채택
            self.path = self._plan_result if self._plan_result is not None else []
            self.idx = 0
            self.phase = "DRIVE"

        # (fallback) 경로가 비었으면(탐색 실패) 안전 정지
        if not self.path:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.3, "gear": "D"}

        # (b) 정차 단계면 계속 브레이크
        if self.phase == "STOP" or reached_goal(state, self.goal):
            self.phase = "STOP"
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # (c) [E] 현재 "방향 세그먼트" 안에서만 진행 — cusp 를 함부로 건너뛰지 않는다.
        v = state.get("v", 0.0)
        seg_dir = _wp_dir(self.path[self.idx])             # 이 세그먼트 전진(+1)/후진(-1)
        seg_end = segment_end(self.path, self.idx)         # 세그먼트 끝(cusp 또는 골)
        # 세그먼트 끝까지만 인덱스를 전진(통과 인정 반경 안의 점들 스킵)
        while self.idx < seg_end:
            wx, wy = _wp_xy(self.path[self.idx])
            if math.hypot(state["x"] - wx, state["y"] - wy) >= PARAMS["reach_radius"]:
                break
            self.idx += 1

        is_cusp = seg_end < len(self.path) - 1             # 이 세그먼트가 전후진 전환으로 끝나나
        ex, ey = _wp_xy(self.path[seg_end])
        d_segend = math.hypot(state["x"] - ex, state["y"] - ey)

        # (c-1) [E] cusp 도달: 먼저 정지한 뒤 다음(반대 방향) 세그먼트로 전환한다.
        if is_cusp and self.idx >= seg_end and d_segend < PARAMS["reach_radius"]:
            if abs(v) > PARAMS["stop_speed"]:
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0,
                        "gear": gear_for_direction(seg_dir)}   # 전환점에서 정지
            self.idx = seg_end + 1                              # 정지 완료 -> 다음 세그먼트
            seg_dir = _wp_dir(self.path[self.idx])
            seg_end = segment_end(self.path, self.idx)

        gear = gear_for_direction(seg_dir)                      # D or R

        # (d) [G] 곡률 적응 속도: 직선이면 빠르게(가속↑), 곡선/주차기동이면 느리게(추종오차↓)
        d_goal = dist_to_goal(state, self.path[-1])            # 골까지 거리
        d_cusp = dist_to_next_cusp(self.path, self.idx)        # 다음 cusp 까지 거리
        v_max = curvature_speed(self.path, self.idx, seg_end)  # 앞쪽 곡률 기반 상한
        # [G] 추종오차 기반 감속: 현재 경로 이탈량(cte)이 클수록 상한을 낮춰 충돌 전 회복 유도
        seg_s = segment_start(self.path, self.idx)
        cte_now = min(math.hypot(state["x"] - _wp_xy(self.path[j])[0],
                                 state["y"] - _wp_xy(self.path[j])[1])
                      for j in range(seg_s, seg_end + 1))
        v_max *= _clamp(1.0 - PARAMS["cte_slow_gain"] * cte_now,
                        PARAMS["cte_slow_floor"], 1.0)
        v_max = max(v_max, PARAMS["v_min"])                     # crawl 최저속도 보장(v_min 아래로 X)
        v_tgt = target_speed(d_goal, d_cusp, self.phase, v_max)  # + cusp/골 감속

        # (e) [E] 조향: 세그먼트 내 동적 look-ahead 점을 향한 Pure Pursuit
        #     (look-ahead 평활 덕에 Hybrid A* 의 지그재그 경로를 안정적으로 추종).
        ld = _clamp(PARAMS["ld_k"] * abs(v) + PARAMS["ld_min"],
                    PARAMS["ld_min"], PARAMS["ld_max"])
        tgt = lookahead_target(self.path, state, self.idx, seg_end, ld)
        steer = pure_pursuit_steer(state, self.path, tgt, seg_dir)  # [C] 조향각
        steer = _clamp(steer, -MAX_STEER, MAX_STEER)               # 한계값으로 자름
        accel, brake = speed_to_pedal(v, v_tgt, seg_dir)
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
