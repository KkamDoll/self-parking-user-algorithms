"""학생 자율주차 알고리즘 스켈레톤 모듈 (경량 버전).

이 파일만 수정하면 되고, 네트워킹/IPC 관련 코드는 `ipc_client.py`에서
자동으로 처리합니다.

[설계] 무거운 Hybrid A*+궤적최적화(scipy)+롤아웃검증(수만 스텝 시뮬)+비동기 스레딩
버전은 계산이 무거워 오히려 응답성이 떨어졌다. 이 버전은 가벼운 파이프라인으로 교체:
  1) set_map: 장애물(점유슬롯/벽/라인)을 격자에 래스터화 + 거리장(BFS) 계산 (1회)
  2) compute_path: 시작/목표 BFS 휴리스틱 + 양방향(meet-in-the-middle) kinematic A* (1회)
  3) compute_control: Stanley(전진)/Pure Pursuit(후진) 추종 (매 스텝)

튜닝 상수는 아래 "튜닝 상수" 블록에 모아두었다 — 여기만 만지면서 조정.
무거운 구버전은 `student_planner_hybrid_1452.bak.py` / git 커밋 33e9303 에 보존.
"""

import math
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


# ===========================================================================
# 튜닝 상수 (여기만 만지면서 성능 조정)
# ===========================================================================
CLEARANCE_CELLS = 5        # 경로탐색용 안전 여유(셀). 실제 충돌은 차체 footprint 검사(car_free)가 막으므로
                           #   이 값은 '대략적 통로'만 정한다. 너무 크면(구 7=3.5m) 주차된 차 사이 통로를
                           #   통째로 막아 경로가 사라진다. 5≈2.5m: 추종오차 여유 + 통로 통과 균형.
YAW_RES = 0.0873           # 탐색 상태 yaw 이산화(rad, ~5°). 작을수록 정밀하나 상태폭증·느림.
                           #   (구버전 0.03rad(~1.7°)은 과도하게 촘촘해 탐색이 느렸음)
STEP_CELLS = 3.5           # 탐색 모션 1스텝 길이(셀). 클수록 탐색은 빠르나 경로가 거칠어짐.
SEARCH_MAX_ITER = 200000   # 경로탐색 반복 상한(응답성 보장). 초과 시 실패 처리 후 정지.

# 속도 프로파일: 목표속도 = 계수 × cell_size (m/s). 거리(check_dist)에 따라 단계적으로.
SPEED_KP = 2.0             # 속도오차 → 페달 P게인. 높을수록 목표속도에 빨리 도달(평균속도↑).
V_FAR   = 13.0             # check_dist > 15셀  : 원거리 순항
V_MID   = 8.0              # check_dist > 10셀
V_NEAR  = 3.0              # check_dist > 5셀
V_SLOW  = 1.5              # check_dist > 1.5셀
V_CRAWL = 0.5              # check_dist > 0.5셀 : 정지 직전 크롤
STOP_DIST = 0.25           # 후축이 목표(슬롯중앙)에 이만큼(m) 들어오면 정지. 작을수록 깊이 들어가나 과주행위험.
ARRIVE_DIST = 0.35         # 후축이 목표에 이만큼(m) 들어오면 '도착'으로 래치 → 완전정지 고정(헌팅/진동 방지).

# 주차 방향 / 조향 평활 (steering reversals 점수 개선용)
ORIENTATION_FREE = True    # 앞뒤 주차방향 무관 → 항상 '전진 드라이브인'이 되는 방향 선택(후진/기어전환 제거).
                           #   ※ 시뮬 STAGE_RULES 의 parking_orientation 가중치(5~7)는 포기하는 대신,
                           #     후진 기동 제거로 steer_flip/time/distance/speed 점수를 더 크게 얻는 트레이드오프.
                           #     expected 방향을 반드시 지켜야 하면 False, 못 푸는 맵이 있어도 False.
STANLEY_K = 2.5            # 전진 Stanley 횡오차 게인 (구 4.0). 낮출수록 좌우 흔들림↓ → 스티어링 전환↓.
STANLEY_SOFT_V = 0.6       # 전진 Stanley 분모 연화속도 (구 0.1). 저속 과민조향 억제 → 흔들림↓.
STEER_SMOOTH = 0.5         # 조향 명령 1차 저역통과 계수(0=이전유지 … 1=즉시반영). 작을수록 부드럽다.
STEER_DEADZONE = 0.0349    # rad(2°). 이 이하 조향은 0으로 스냅 → 직진 부근 미세 좌우떨림(=flip) 제거
                           #   (시뮬 STEER_FLIP_DEADZONE 과 동일값).

# 차량 footprint 충돌검사 (실제 시뮬 충돌 = 경계 + 점유슬롯 SAT + 라인 SAT)
CAR_LF = 1.6               # 후축→앞 오버행 (sim Params.Lf)
CAR_LR = 1.4               # 후축→뒤 오버행 (sim Params.Lr)
CAR_W  = 1.6               # 차폭 (sim Params.W)
CAR_INFLATE = 0.08         # footprint 안전여유(m, 추종오차 흡수). 키우면 안전하나 좁은 슬롯 진입 실패↑.
LINE_HALF = 0.25           # painted line 반폭(m) (sim 라인 충돌과 동일)


def _obb_hits_rect(corners, c, s, rx0, rx1, ry0, ry1):
    """회전 차량 OBB(corners) vs 축정렬 사각형(rx0..ry1) SAT 충돌 판정."""
    xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
    if max(xs) < rx0 or min(xs) > rx1 or max(ys) < ry0 or min(ys) > ry1:
        return False  # AABB 분리 → 사각형의 두 축(world x,y)으로 분리됨
    rect = ((rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1))
    for ux, uy in ((c, s), (-s, c)):   # 차량의 두 축만 추가 검사
        cmn = cmx = corners[0][0]*ux + corners[0][1]*uy
        for px, py in corners[1:]:
            d = px*ux + py*uy
            if d < cmn: cmn = d
            elif d > cmx: cmx = d
        rmn = rmx = rect[0][0]*ux + rect[0][1]*uy
        for px, py in rect[1:]:
            d = px*ux + py*uy
            if d < rmn: rmn = d
            elif d > rmx: rmx = d
        if cmx < rmn or rmx < cmn:
            return False
    return True


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
    base_board = None
    collision_map = None
    cur_idx = 0
    is_reverse = False
    prev_steer = 0.0       # 직전 스텝 조향(저역통과 평활용)
    arrived = False        # 목표 도착 래치(도달 후 완전정지 고정 → 진동/이탈 방지)
    _plan_failed = False   # 경로탐색 실패 래치(매 틱 재계산 폭주 방지)

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
        self.parked_grid = map_payload.get("grid", {}).get("parked")
        self.slots = map_payload.get("slots")
        self.occupied_idx = map_payload.get("occupied_idx")
        self.walls = map_payload.get("walls_rects")
        self.lines = map_payload.get("lines")
        pretty_print_map_summary(map_payload)
        self.waypoints.clear()
        self.collision_map = None
        self.cur_idx = 0
        self.is_reverse = False
        self.prev_steer = 0.0
        self.arrived = False
        self._plan_failed = False
        self.expected_orientation = map_payload["expected_orientation"]

        self.base_board = [[0.0]*len(self.parked_grid[0]) for _ in range(len(self.parked_grid))]
        self.collision_map = [[0.0]*len(self.parked_grid[0]) for _ in range(len(self.parked_grid))]
        q = deque()

        def transform_idx(x_min, x_max, y_min, y_max):
            x_min -= self.map_extent[0]
            x_max -= self.map_extent[0]
            y_max, y_min = self.map_extent[3]-y_min, self.map_extent[3]-y_max
            x_min /= self.cell_size
            x_max /= self.cell_size
            y_min /= self.cell_size
            y_max /= self.cell_size
            return map(round, (x_min, x_max, y_min, y_max))

        n_rows = len(self.base_board)
        n_cols = len(self.base_board[0])

        def mark_rect(x_min, x_max, y_min, y_max):
            for i in range(y_min-1, y_max):
                for j in range(x_min, x_max+1):
                    if i < 0 or j >= n_cols:
                        continue
                    self.base_board[i][j] = 1.0
                    self.collision_map[i][j] = 1.0
                    q.append((j, i))  # 장애물 좌표 (x, y)

        ## occupied_slots
        for i in range(len(self.slots)):
            if self.occupied_idx[i]:
                x_min, x_max, y_min, y_max = self.slots[i]
                x_min, x_max, y_min, y_max = transform_idx(x_min, x_max, y_min, y_max)
                mark_rect(x_min, x_max, y_min, y_max)

        ## walls
        for x_min, x_max, y_min, y_max in self.walls:
            x_min, x_max, y_min, y_max = transform_idx(x_min, x_max, y_min, y_max)
            mark_rect(x_min, x_max, y_min, y_max)

        ## lines
        for x_min, y_min, x_max, y_max in self.lines:
            x_min, x_max, y_min, y_max = transform_idx(x_min, x_max, y_min, y_max)
            mark_rect(x_min, x_max, y_min, y_max)

        ## 거리장(BFS): collision_map[y][x] = 가장 가까운 장애물까지의 거리(+1), CLEARANCE_CELLS 에서 상한.
        ##   값 0  = 어떤 장애물로부터도 (CLEARANCE_CELLS-1)셀 이상 떨어진 '주행 가능' 셀.
        ##   값 !=0 = 장애물이거나 그 안전여유 안 → 주행 금지.
        while q:
            cx, cy = q.popleft()
            current_dist = self.collision_map[cy][cx]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < n_cols and 0 <= ny < n_rows:
                    if self.collision_map[ny][nx] == 0:
                        self.collision_map[ny][nx] = current_dist + 1
                        if self.collision_map[ny][nx] == CLEARANCE_CELLS:
                            continue
                        q.append((nx, ny))


## ==============================================          compute_path             ==============================================================================================
    def compute_path(self, obs: Dict[str, Any]) -> None:
        """관측과 맵을 이용해 경로(웨이포인트)를 준비합니다.

        시작점에서 시작한 전방 트리와 목표점에서 시작한 (후방) 트리를 동시에 키워
        같은 이산 상태(x_idx, y_idx, yaw_idx)에서 만나면 잇는 양방향 kinematic A*.
        휴리스틱은 상대편 격자 BFS 거리장.
        """
        self.waypoints.clear()

        def to_idx(x, y):
            return round((x - self.map_extent[0])/self.cell_size), round((self.map_extent[3] - y)/self.cell_size)

        ## 관측값 저장
        x = obs["state"]["x"]
        y = obs["state"]["y"]
        yaw = obs["state"]["yaw"]
        v = obs["state"]["v"]
        x1, x2, y1, y2 = obs["target_slot"]
        dt = obs["limits"]["dt"]
        L = obs["limits"]["L"]
        maxSteer = obs["limits"]["maxSteer"]
        maxAccel = obs["limits"]["maxAccel"]
        maxBrake = obs["limits"]["maxBrake"]
        steerRate = obs["limits"]["steerRate"]

        n_rows = len(self.base_board)
        n_cols = len(self.base_board[0])

        target_middle_x_idx, target_middle_y_idx = to_idx((x1+x2)/2, (y1+y2)/2)

        # 진입 방향 결정. ORIENTATION_FREE 면 차가 '전진 드라이브인(cusp 없음·후진 안 함)'이 되는
        # 방향을 고른다. 실측 결과: 차는 아래에서 위(+y)를 보고 출발하므로, 멀리(위쪽) 슬롯은 front_in,
        # 가까운(아래쪽) 슬롯은 rear_in 으로 잡아야 후진 진입(back-in)이 사라진다. → 스티어링 좌우전환 급감.
        # (주의: 시뮬 orientation 라벨은 최종 헤딩 부호로만 정해지므로 가까운 슬롯은 'rear_in' 라벨이지만
        #  실제 기동은 전진 드라이브인이다 — 사용자가 싫어한 건 '후진으로 들어가는 기동'.)
        eff_orientation = self.expected_orientation
        if ORIENTATION_FREE:
            eff_orientation = "front_in" if target_middle_y_idx < n_rows/3 else "rear_in"

        forward = True
        if eff_orientation == 'front_in':
            if target_middle_y_idx < n_rows/3:
                forward = False
        else:
            if target_middle_y_idx > n_rows/3:
                forward = False

        target_x = (x1 + x2) / 2
        # 차체를 슬롯 '중앙'에 두도록 후축 목표를 잡는다(완전 포함 → 라인/경계 충돌 면제).
        # 후축은 차체중심보다 body_off=(Lf-Lr)/2=0.1m 뒤이므로 헤딩 반대로 그만큼 민다.
        # (구: ±L/4=0.65 는 차를 한쪽으로 치우치게 해 반대쪽 코너가 슬롯/벽/라인 밖으로 삐져나가 충돌.)
        body_off = (CAR_LF - CAR_LR) / 2.0
        if eff_orientation == "front_in":   # nose +y
            target_y = (y1 + y2) / 2 - body_off
        else:                               # nose -y
            target_y = (y1 + y2) / 2 + body_off

        target_hmap = [[-1] * n_cols for _ in range(n_rows)]  ## target_pnt 로부터의 BFS 거리
        target_x_idx, target_y_idx = to_idx(target_x, target_y)
        # 목표점은 슬롯/장애물 안전여유 안에 있어 막혀 있다 → 위/아래로 채널을 뚫어 도달 가능하게 함.
        i, j = target_y_idx, target_x_idx
        while self.collision_map[i][j] != 0:
            self.collision_map[i][j] = 0
            if i < n_rows/3:
                i += 1
            else:
                i -= 1

        target_hmap[target_y_idx][target_x_idx] = 0
        q = deque([(target_x_idx, target_y_idx)])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < n_cols and 0 <= ny < n_rows:
                    if target_hmap[ny][nx] == -1 and self.collision_map[ny][nx] == 0:
                        target_hmap[ny][nx] = target_hmap[cy][cx] + self.cell_size
                        q.append((nx, ny))

        start_hmap = [[-1] * n_cols for _ in range(n_rows)]  ## start_pnt 로부터의 BFS 거리
        start_x, start_y = x, y
        start_x_idx, start_y_idx = to_idx(x, y)
        start_hmap[start_y_idx][start_x_idx] = 0
        q = deque([(start_x_idx, start_y_idx)])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < n_cols and 0 <= ny < n_rows:
                    if start_hmap[ny][nx] == -1 and self.collision_map[ny][nx] == 0:
                        start_hmap[ny][nx] = start_hmap[cy][cx] + self.cell_size
                        q.append((nx, ny))

        # ---- 차량 footprint 충돌검사 준비 (탐색 노드를 점이 아니라 '차체'로 검사) ----
        # 실제 시뮬 충돌 기하: 경계(extent) + 점유 옆슬롯 + painted line. 단, '목표 슬롯'에 겹치는
        # 라인은 제외해야 차가 슬롯 안으로 진입할 수 있다(슬롯 안에 들어가면 라인 충돌 면제).
        infl = CAR_INFLATE
        _fL, _bL, _wH = CAR_LF + infl, CAR_LR + infl, CAR_W/2 + infl
        car_local = ((_fL, _wH), (_fL, -_wH), (-_bL, -_wH), (-_bL, _wH))
        xmin_w, xmax_w, ymin_w, ymax_w = self.map_extent
        tgt_rect = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        obs_rects = []
        for s_i in range(len(self.slots)):                  # 점유된 옆 슬롯(주차된 차)
            if self.occupied_idx[s_i]:
                sx0, sx1, sy0, sy1 = (float(t) for t in self.slots[s_i])
                obs_rects.append((min(sx0, sx1), max(sx0, sx1), min(sy0, sy1), max(sy0, sy1)))
        for seg in self.lines:                              # painted line → 반폭 사각형
            lx1, ly1, lx2, ly2 = (float(t) for t in seg)
            if abs(lx1 - lx2) < 1e-6:                        # 수직선
                a, b, c2, d2 = min(lx1, lx2)-LINE_HALF, max(lx1, lx2)+LINE_HALF, min(ly1, ly2), max(ly1, ly2)
            elif abs(ly1 - ly2) < 1e-6:                      # 수평선
                a, b, c2, d2 = min(lx1, lx2), max(lx1, lx2), min(ly1, ly2)-LINE_HALF, max(ly1, ly2)+LINE_HALF
            else:
                a, b, c2, d2 = (min(lx1, lx2)-LINE_HALF, max(lx1, lx2)+LINE_HALF,
                                min(ly1, ly2)-LINE_HALF, max(ly1, ly2)+LINE_HALF)
            if not (b < tgt_rect[0] or a > tgt_rect[1] or d2 < tgt_rect[2] or c2 > tgt_rect[3]):
                continue                                     # 목표 슬롯과 겹치는 라인은 장애물에서 제외
            obs_rects.append((a, b, c2, d2))

        def car_free(cx, cy, cyaw):
            cs = math.cos(cyaw); sn = math.sin(cyaw)
            corners = [(cx + lx*cs - ly*sn, cy + lx*sn + ly*cs) for lx, ly in car_local]
            xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
            if min(xs) < xmin_w or max(xs) > xmax_w or min(ys) < ymin_w or max(ys) > ymax_w:
                return False                                 # 경계 이탈
            for rx0, rx1, ry0, ry1 in obs_rects:
                if _obb_hits_rect(corners, cs, sn, rx0, rx1, ry0, ry1):
                    return False
            return True

        class Node:
            __slots__ = ("x", "y", "yaw", "x_idx", "y_idx", "yaw_idx", "steer", "direction", "cost", "kind", "parent")
            def __init__(self, x, y, yaw, x_idx, y_idx, yaw_idx, steer, direction, cost, kind, parent):
                self.x = x
                self.y = y
                self.yaw = yaw
                self.x_idx = x_idx
                self.y_idx = y_idx
                self.yaw_idx = yaw_idx
                self.steer = steer
                self.direction = direction
                self.cost = cost
                self.kind = kind  ## 시작점부터 시작한 노드 or 목적지부터 시작한 노드
                self.parent = parent

            def __lt__(self, other):
                return self.cost < other.cost

        start_yaw_idx = round(yaw / YAW_RES)
        start_node = Node(start_x, start_y, yaw, start_x_idx, start_y_idx, start_yaw_idx, 0.0, 1, 0.0, "start", None)

        target_yaw = yaw if eff_orientation == "front_in" else -yaw
        target_yaw_idx = round(target_yaw / YAW_RES)
        target_node = Node(target_x, target_y, target_yaw, target_x_idx, target_y_idx, target_yaw_idx, 0.0, 1, 0.0, "target", None)
        if not forward:
            target_node.direction = -1
        pq = [(target_hmap[start_y_idx][start_x_idx], start_node), (start_hmap[target_y_idx][target_x_idx], target_node)]

        visited = {}
        visited[(start_x_idx, start_y_idx, start_yaw_idx)] = start_node
        visited[(target_x_idx, target_y_idx, target_yaw_idx)] = target_node

        steer_inputs = [-maxSteer, -maxSteer/2, 0, maxSteer/2, maxSteer]

        for _ in range(SEARCH_MAX_ITER):
            if not pq:
                break
            f, node = heapq.heappop(pq)
            for next_steer in steer_inputs:
                if abs(next_steer-node.steer) > maxSteer:
                    continue
                next_direction = node.direction
                distance = next_direction * self.cell_size * STEP_CELLS

                if abs(next_steer) < 0.001:  ## 직진
                    next_yaw = node.yaw
                    next_x = node.x + distance * math.cos(node.yaw)
                    next_y = node.y + distance * math.sin(node.yaw)
                else:  ## 회전
                    beta = (distance / L) * math.tan(next_steer)
                    R = L / math.tan(next_steer)
                    next_yaw = node.yaw + beta
                    next_x = node.x + R * (math.sin(next_yaw) - math.sin(node.yaw))
                    next_y = node.y - R * (math.cos(next_yaw) - math.cos(node.yaw))
                    # 각도 정규화 (-pi ~ pi)
                    next_yaw = (next_yaw + math.pi) % (2 * math.pi) - math.pi
                if node.kind == 'target':
                    if abs(node.yaw) < abs(next_yaw):
                        continue
                next_yaw_idx = round(next_yaw / YAW_RES)
                next_x_idx, next_y_idx = to_idx(next_x, next_y)
                if not (0 <= next_x_idx < n_cols and 0 <= next_y_idx < n_rows):
                    continue

                ## 차체 footprint 충돌검사 (점이 아니라 실제 차체로 옆슬롯/라인/경계 검사)
                if not car_free(next_x, next_y, next_yaw):
                    continue

                ## 이미 지나간 경로인지 확인
                if (next_x_idx, next_y_idx, next_yaw_idx) in visited:
                    ## 시작점에서 시작한 노드와 목적지에서 시작한 노드가 만나면 종료
                    matched_node = visited[(next_x_idx, next_y_idx, next_yaw_idx)]

                    if matched_node.kind != node.kind:
                        first_node, final_node = matched_node, node
                        if node.kind == "start":
                            first_node, final_node = node, matched_node
                        path = []
                        while first_node:
                            path.append((first_node.x, first_node.y))
                            first_node = first_node.parent
                        path = path[::-1]
                        while final_node:
                            path.append((final_node.x, final_node.y))
                            final_node = final_node.parent
                        self.waypoints = path

                        # control 함수 돌리기 전 전처리
                        self.cur_idx = 0
                        self.is_reverse = False
                        return
                    continue

                new_h = 999
                if node.kind == "start":
                    if target_hmap[next_y_idx][next_x_idx] < 0:
                        continue
                    new_h = target_hmap[next_y_idx][next_x_idx]
                else:
                    if start_hmap[next_y_idx][next_x_idx] < 0:
                        continue
                    new_h = start_hmap[next_y_idx][next_x_idx]
                new_h += abs(next_yaw)

                step_cost = abs(distance)

                if abs(next_steer - node.steer) > 0.001:
                    step_cost += abs(distance)*3  ## 핸들조작 패널티

                new_cost = node.cost + step_cost
                next_node = Node(next_x, next_y, next_yaw, next_x_idx, next_y_idx, next_yaw_idx, next_steer, next_direction, new_cost, node.kind, node)

                visited[(next_x_idx, next_y_idx, next_yaw_idx)] = next_node

                heapq.heappush(pq, (new_h+new_cost, next_node))

## =========================================================================================================================================================================


    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """경로를 따라가기 위한 조향/가감속 명령을 산출합니다."""
        if not self.waypoints:
            if self._plan_failed:
                # 이미 경로탐색에 실패함 → 매 틱 재계산(폭주) 대신 정지 유지.
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}
            self.compute_path(obs)
            if not self.waypoints:
                self._plan_failed = True
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # OBS 불러오기
        v = float(obs.get("state", {}).get("v", 0.0))
        x = obs["state"]["x"]
        y = obs["state"]["y"]
        yaw = obs["state"]["yaw"]
        dt = obs["limits"]["dt"]
        L = obs["limits"]["L"]
        maxSteer = obs["limits"]["maxSteer"]
        maxAccel = obs["limits"]["maxAccel"]
        maxBrake = obs["limits"]["maxBrake"]

        # 도착 래치: 한번 목표에 도달하면 끝까지 완전정지 고정(속도 헌팅으로 |v|>0.2 가 풀리는 것 방지).
        if self.arrived:
            self.prev_steer = 0.0
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        # 조향각 구하기 Stanley 알고리즘 사용

        target_x, target_y = self.waypoints[-1]  # 후륜 지점.

        last_x, last_y = self.waypoints[-1]
        prev_x, prev_y = self.waypoints[-2]
        heading = math.atan2(last_y - prev_y, last_x - prev_x)

        # 목표지점 전륜 좌표.
        target_x_front = target_x + L * math.cos(heading) * 1.5
        target_y_front = target_y + L * math.sin(heading) * 1.5

        # 현재 좌표 전륜 위치 계산
        front_x = x + L * math.cos(yaw)
        front_y = y + L * math.sin(yaw)

        # 기어번경 지점 탐색
        switch_idx = -1
        dist_to_switch = float('inf')
        search_limit = min(self.cur_idx + 20, len(self.waypoints) - 1)

        for i in range(self.cur_idx, search_limit):
            if i + 2 < len(self.waypoints):
                curr_p = self.waypoints[i]
                next_p = self.waypoints[i+1]
                yaw1 = math.atan2(next_p[1] - curr_p[1], next_p[0] - curr_p[0])
                next_next_p = self.waypoints[i+2]
                yaw2 = math.atan2(next_next_p[1] - next_p[1], next_next_p[0] - next_p[0])
                diff = abs(yaw1 - yaw2)
                diff = (diff + math.pi) % (2*math.pi) - math.pi
                if abs(diff) > math.pi / 2:
                    switch_idx = i+1
                    dist_to_switch = math.hypot(x - self.waypoints[switch_idx][0], y - self.waypoints[switch_idx][1])
                    break

        # 가까운 경로 찾기
        min_dist = float('inf')
        start_search = self.cur_idx
        end_search = min(len(self.waypoints), self.cur_idx + 20)

        if switch_idx != -1 and dist_to_switch > 0.5 * self.cell_size:
            if self.cur_idx == switch_idx - 1:
                min_dist = self.cell_size
            end_search = min(end_search, switch_idx + 1)
        for i in range(start_search, end_search):
            wx, wy = self.waypoints[i]
            dist = math.hypot(x - wx, y - wy)
            if dist < min_dist:
                min_dist = dist
                self.cur_idx = i

        # 조향각 계산
        tx, ty = self.waypoints[self.cur_idx]

        next_idx = self.cur_idx + 1
        if switch_idx != -1 and next_idx > switch_idx:
            next_idx = switch_idx

        if next_idx < len(self.waypoints):
            nx, ny = self.waypoints[next_idx]
            path_yaw = math.atan2(ny - ty, nx - tx)
        else:
            nx, ny = tx, ty
            px, py = self.waypoints[self.cur_idx - 1]
            path_yaw = math.atan2(ty - py, tx - px)

        angle_to_path = path_yaw - yaw
        angle_to_path = (angle_to_path + math.pi) % (2 * math.pi) - math.pi

        # 후진 조향각 처리
        if not self.is_reverse:
            if switch_idx != -1 and dist_to_switch > 0.5 * self.cell_size:
                self.is_reverse = False
            else:
                self.is_reverse = abs(angle_to_path) > math.pi / 2

        # 전진/후진 모두 '후축 → 후축목표' 거리로 정지/감속 판단.
        # (구: 전진은 전륜을 1.5L 앞 점에 맞춰 0.5L≈1.3m 과주행 → 벽/라인에 박았다.)
        dist_to_goal = math.hypot(target_x - x, target_y - y)

        # 목표 도달 → 도착 래치 후 완전정지(이후 매 스텝 위 early-return으로 고정).
        if dist_to_goal < ARRIVE_DIST:
            self.arrived = True
            self.prev_steer = 0.0
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        if self.is_reverse:
            ld_base = 1.5
            ld_gain = 0.5
            ld = ld_base + ld_gain * abs(v)

            extension_dist = 5.0
            ext_x = last_x + extension_dist * math.cos(heading)
            ext_y = last_y + extension_dist * math.sin(heading)  # (수정) 기존 cos → sin

            # Look-ahead Point 찾기
            # 현재 인덱스부터 경로를 따라가며 ld보다 멀리 있는 점을 찾음
            target_pt = (ext_x, ext_y)

            for i in range(self.cur_idx, len(self.waypoints)):
                wx, wy = self.waypoints[i]
                d = math.hypot(wx - x, wy - y)
                if d > ld:
                    target_pt = (wx, wy)
                    break

            t_px, t_py = target_pt

            # Alpha 계산 (차량 후방 기준)
            rear_yaw = yaw + math.pi

            # 타겟을 바라보는 각도
            target_angle = math.atan2(t_py - y, t_px - x)

            # 후방 벡터와 타겟 벡터 사이의 각도 차이
            alpha = target_angle - rear_yaw
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

            # Pure Pursuit 공식 적용
            # delta = atan(2 * L * sin(alpha) / Ld)
            new_steer = -math.atan2(2.0 * L * math.sin(alpha), ld)

        else:
            # waypoint 후륜좌표를 전륜 좌표로 변경
            next_front_x = tx + L * math.cos(path_yaw)
            next_front_y = ty + L * math.sin(path_yaw)

            # stanley 알고리즘
            cte = -math.sin(path_yaw) * (front_x - next_front_x) + \
                math.cos(path_yaw) * (front_y - next_front_y)
            # 전진 파라미터
            k = STANLEY_K
            soft_v = STANLEY_SOFT_V
            new_steer = angle_to_path - math.atan2(k * cte, max(abs(v), soft_v))

        new_steer = new_steer if maxSteer > abs(new_steer) else maxSteer if new_steer > 0 else -maxSteer

        # 조향 평활 + 데드존: 좌우로 핸들이 자주 뒤집히는 것(steering reversals 점수↓ 원인)을 억제.
        new_steer = self.prev_steer + STEER_SMOOTH * (new_steer - self.prev_steer)  # 1차 저역통과
        if abs(new_steer) < STEER_DEADZONE:                                          # 직진 부근 떨림 제거
            new_steer = 0.0

        # 속도 제어 (P 제어기)

        check_dist = dist_to_goal

        if switch_idx != -1:
            check_dist = dist_to_switch

        if check_dist > 15.0*self.cell_size:
            target_v = V_FAR*self.cell_size
        elif check_dist > 10.0*self.cell_size:
            target_v = V_MID*self.cell_size
        elif check_dist > 5.0*self.cell_size:
            target_v = V_NEAR*self.cell_size
        elif check_dist > 1.5*self.cell_size:
            target_v = V_SLOW*self.cell_size
        elif check_dist > 0.5*self.cell_size:
            target_v = V_CRAWL*self.cell_size
        else:
            target_v = 0.0

        current_v = v
        speed_error = target_v - abs(current_v)

        cmd = SPEED_KP * speed_error

        accel = 0.0
        brake = 0.0

        if cmd > 0:
            accel = min(cmd, maxAccel)
            brake = 0.0
        else:
            brake = -cmd
            accel = 0.0
            brake = min(brake, maxBrake)

        if self.is_reverse:
            if v > 0.1:
                accel = 0.0
                brake = maxBrake
        else:
            if v < -0.1:
                accel = 0.0
                brake = maxBrake

        if self.is_reverse:
            if dist_to_goal < STOP_DIST:
                accel = 0.0
                brake = maxBrake
                new_steer = 0.0
        else:
            if dist_to_goal < STOP_DIST:
                accel = 0.0
                brake = maxBrake

        self.prev_steer = new_steer
        return {
            "steer": float(new_steer),
            "accel": float(accel),
            "brake": float(brake),
            "gear": "R" if self.is_reverse else "D"
        }


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
