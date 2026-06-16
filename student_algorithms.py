"""학생 자율주차 알고리즘 클라이언트.

이 스크립트는 시뮬레이터(`demo_self_parking_sim.py`)가 개방한 TCP 포트에
JSONL 프로토콜로 접속한다. 통신 흐름은 아래와 같다.

0. 연결이 성립하면 시뮬레이터가 정적 맵(`{"map": ...}`)을 먼저 보낸다.
1. 이후 시뮬레이터가 매 스텝마다 관측 패킷(`obs`)을 보낸다. 주요 필드:
   - `t`: 시뮬레이터 시간이 초 단위로 증가
   - `state`: 현재 차량 위치(x, y), 각도(yaw), 속도(v)
   - `target_slot`: 목표 주차 슬롯의 직사각형 좌표(xmin, xmax, ymin, ymax)
   - `limits`: 차량 제한(타임스텝, 휠베이스, 조향/가감속 한계 등)
2. 학생 알고리즘은 이 정보를 이용해 다음 명령(`cmd`)을 계산하고,
   `{"steer", "accel", "brake", "gear"}` 값을 JSON 한 줄로 응답한다.
3. 시뮬레이터는 받은 명령을 차량 모델에 적용하고 다음 관측을 보낸다.

학생은 `planner_step()`을 원하는 로직으로 수정해 관측→명령 계산을 구현하면 된다.
시뮬레이터가 아직 실행 중이 아니면 자동으로 재시도하며, 연결이 끊어지면 즉시
다시 접속을 시도한다.
"""

import argparse
import heapq
import json
import math
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _wrap(angle: float) -> float:
    """각도를 -pi ~ +pi 범위로 정규화."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PlannerSkeleton:
    """주차 경로 계획 + 추종.

    [담당: 맵 분석 & 경로 계획]
      1) set_map()    : 맵 데이터를 파싱해 '장애물 격자(OCC)'를 만든다.
      2) compute_path(): 차량 운동 제약(전/후진 + 조향)을 고려한 Hybrid A* 로
                         목표 슬롯까지 충돌 없는 경로(waypoints)를 생성한다.
      3) compute_control(): 생성한 경로를 따라가는 '참고용' 추종 제어.
                         (※ 실제 제어 고도화는 제어 담당이 이어받음)

    왜 일반 A* 가 아니라 Hybrid A* 인가?
      - 일반 A* 는 격자 칸을 상하좌우로 점프한다. 차는 옆으로 못 가고(비홀로노믹),
        조향 반경 제약이 있어서 그 경로는 실제로 못 따라간다.
      - Hybrid A* 는 노드를 (x, y, yaw) 연속 상태로 두고, 실제 자전거 모델로
        '전진/후진 × 조향각' 만큼 움직인 결과를 자식 노드로 확장한다.
        그래서 나온 경로가 곧바로 주행 가능하다.
    """

    # ---- 차량/탐색 파라미터 (시뮬레이터 값과 맞춤) ----
    WB = 2.6                       # 휠베이스(wheelbase)
    CAR_F = 1.6                    # 기준점에서 앞범퍼까지(Lf)
    CAR_R = 1.4                    # 기준점에서 뒷범퍼까지(Lr)
    CAR_W = 1.6                    # 차폭
    MAX_STEER = math.radians(35)   # 최대 조향각
    MARGIN = 0.15                  # 충돌 안전 여유(m)

    STEERS = [-MAX_STEER, -MAX_STEER / 2, 0.0, MAX_STEER / 2, MAX_STEER]
    DIRS = [1.0, -1.0]             # +1 전진, -1 후진
    DS = 1.0                       # 한 번 확장할 호 길이(m)
    SUB = 4                        # 호를 몇 등분해 충돌검사할지
    XY_RES = 0.5                   # 방문 격자 해상도(중복 노드 제거용)
    YAW_RES = math.radians(15)     # yaw 격자 해상도
    H_WEIGHT = 1.2                 # 가중 A*(클수록 빠르지만 덜 최적)

    POS_TOL = 0.6                  # 목표 도달 판정 거리(m)
    YAW_TOL = math.radians(25)     # 목표 도달 판정 각도
    MAX_POPS = 120000              # 무한루프 방지용 탐색 상한

    def __init__(self) -> None:
        self.map_data: Optional[Dict[str, Any]] = None
        self.OCC: Optional[np.ndarray] = None      # True=장애물
        self.extent = (0.0, 0.0, 0.0, 0.0)         # (xmin,xmax,ymin,ymax)
        self.cellSize = 0.5
        self.H = 0
        self.W = 0
        self.waypoints: List[Tuple[float, float, float, float]] = []  # (x,y,yaw,dir)
        self._planned = False

    # =====================================================================
    # 1) 맵 파싱
    # =====================================================================
    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """시뮬레이터가 1회 보내주는 정적 맵을 파싱.

        받는 키: extent[xmin,xmax,ymin,ymax], cellSize, slots, occupied_idx,
                 walls_rects, grid{stationary, parked}
        만드는 것: OCC(장애물 격자). 벽(stationary)과 주차된 차(parked)를
                   합쳐 'True=막힘' 으로 표시한다. (값 1=막힘, 0=빈공간)
        """
        self.map_data = map_payload
        self.extent = tuple(float(v) for v in map_payload["extent"])
        self.cellSize = float(map_payload["cellSize"])

        grid = map_payload.get("grid", {})
        cs = np.asarray(grid.get("stationary", []), dtype=float)
        cp = np.asarray(grid.get("parked", []), dtype=float)
        # 두 격자를 합쳐 장애물 마스크 생성 (둘 중 하나라도 막혀 있으면 장애물)
        occ = np.zeros_like(cs, dtype=bool)
        if cs.size:
            occ |= cs > 0.5
        if cp.size:
            occ |= cp > 0.5
        self.OCC = occ
        self.H, self.W = occ.shape if occ.size else (0, 0)
        self._planned = False
        self.waypoints = []

    # ---- 좌표 변환: 월드(x,y) -> 격자(row,col) ----
    # 시뮬레이터와 동일 규칙: row 0 이 맨 위(ymax), 아래로 갈수록 row 증가
    def _rc(self, x: float, y: float) -> Tuple[int, int]:
        xmin, xmax, ymin, ymax = self.extent
        col = int((x - xmin) / self.cellSize)
        row = int((ymax - y) / self.cellSize)
        return row, col

    # ---- 차량 footprint 충돌 검사 ----
    # 차 외곽 직사각형 내부를 0.3m 간격으로 찍어 장애물 칸에 닿는지 본다.
    def _car_free(self, x: float, y: float, yaw: float) -> bool:
        xmin, xmax, ymin, ymax = self.extent
        c, s = math.cos(yaw), math.sin(yaw)
        dl = -self.CAR_R - self.MARGIN
        while dl <= self.CAR_F + self.MARGIN + 1e-6:
            dw = -self.CAR_W / 2 - self.MARGIN
            while dw <= self.CAR_W / 2 + self.MARGIN + 1e-6:
                px = x + dl * c - dw * s
                py = y + dl * s + dw * c
                if not (xmin <= px <= xmax and ymin <= py <= ymax):
                    return False
                r, co = self._rc(px, py)
                if 0 <= r < self.H and 0 <= co < self.W and self.OCC[r, co]:
                    return False
                dw += 0.3
            dl += 0.3
        return True

    # =====================================================================
    # 2) 경로 계획: Hybrid A*
    # =====================================================================
    def _build_hfield(self, gx: float, gy: float) -> Optional[np.ndarray]:
        """목표 셀에서 출발하는 Dijkstra 거리장(휴리스틱).

        장애물을 '피해서' 잰 실제 거리라서, 단순 직선거리보다 훨씬 똑똑하다.
        덕분에 Hybrid A* 탐색이 수십 배 빨라진다(맵 끝~끝 15초 -> 1초 미만).
        """
        if self.OCC is None or not self.OCC.size:
            return None
        free = ~self.OCC
        gr, gc = self._rc(gx, gy)
        if not (0 <= gr < self.H and 0 <= gc < self.W and free[gr, gc]):
            return None
        INF = 1e9
        dist = np.full((self.H, self.W), INF, dtype=float)
        dist[gr, gc] = 0.0
        pq = [(0.0, gr, gc)]
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist[r, c]:
                continue
            for dr, dc, w in nbrs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and free[nr, nc]:
                    nd = d + w
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(pq, (nd, nr, nc))
        return dist * self.cellSize

    def _disc(self, x: float, y: float, yaw: float) -> Tuple[int, int, int]:
        """연속 상태 (x,y,yaw) 를 격자 인덱스로 이산화 (방문 중복 제거 키)."""
        return (int(round(x / self.XY_RES)),
                int(round(y / self.XY_RES)),
                int(round(_wrap(yaw) / self.YAW_RES)))

    def _expand(self, x: float, y: float, yaw: float, steer: float, d: float):
        """자전거 모델로 (steer, 방향d) 만큼 호를 따라 이동. 중간 충돌이면 None."""
        seg = self.DS / self.SUB
        for _ in range(self.SUB):
            x += d * seg * math.cos(yaw)
            y += d * seg * math.sin(yaw)
            yaw = _wrap(yaw + d * seg / self.WB * math.tan(steer))
            if not self._car_free(x, y, yaw):
                return None
        return x, y, yaw

    def compute_path(self, obs: Dict[str, Any]) -> None:
        """현재 차량 상태 -> 목표 슬롯 중심까지 Hybrid A* 경로 생성.

        결과는 self.waypoints = [(x, y, yaw, dir), ...] 로 저장.
        dir(+1 전진 / -1 후진)은 제어 담당이 기어 결정에 사용한다.
        경로 계획은 '시작 시 한 번만' 수행한다(매 스텝 재계획은 비쌈).
        """
        st = obs["state"]
        sx, sy, syaw = float(st["x"]), float(st["y"]), float(st["yaw"])

        xmin, xmax, ymin, ymax = obs["target_slot"]
        gx = 0.5 * (xmin + xmax)
        gy = 0.5 * (ymin + ymax)
        # 이 주차장은 슬롯이 가로 밴드로 배치 -> 주차 시 차는 세로(yaw=±90°).
        # 두 방향 모두 허용하면 탐색이 도달 가능한 쪽으로 알아서 끝난다.
        goal_yaws = [math.radians(90), math.radians(-90)]

        hf = self._build_hfield(gx, gy)

        def heur(x: float, y: float) -> float:
            if hf is not None:
                r, c = self._rc(x, y)
                if 0 <= r < self.H and 0 <= c < self.W and hf[r, c] < 1e8:
                    return float(hf[r, c])
            return math.hypot(gx - x, gy - y)   # 거리장 밖이면 직선거리 fallback

        start = (sx, sy, syaw)
        open_heap: List[Tuple[float, int, Tuple[float, float, float], float, float, float]] = []
        counter = 0
        heapq.heappush(open_heap, (heur(sx, sy), counter, start, 0.0, -1.0, 0.0))
        counter += 1
        came_from: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
        g_cost: Dict[Tuple[int, int, int], float] = {self._disc(*start): 0.0}
        pops = 0
        found = None

        while open_heap:
            _, _, node, g, pdir, psteer = heapq.heappop(open_heap)
            pops += 1
            if pops > self.MAX_POPS:
                break
            x, y, yaw = node
            # 목표 판정: 위치가 가깝고 + 세로 방향(±90°) 근처면 도착
            if (math.hypot(gx - x, gy - y) < self.POS_TOL and
                    min(abs(_wrap(yaw - q)) for q in goal_yaws) < self.YAW_TOL):
                found = node
                break
            for d in self.DIRS:
                for steer in self.STEERS:
                    nxt = self._expand(x, y, yaw, steer, d)
                    if nxt is None:
                        continue
                    nk = self._disc(*nxt)
                    # 비용: 거리 + (후진/방향전환/조향) 패널티 -> 자연스러운 경로 유도
                    cost = self.DS * (1.0 if d > 0 else 1.5)
                    if pdir != -1.0 and d != pdir:
                        cost += 2.0                       # 전/후진 전환 패널티
                    cost += 0.3 * abs(steer) + 0.2 * abs(steer - psteer)
                    ng = g + cost
                    if nk not in g_cost or ng < g_cost[nk]:
                        g_cost[nk] = ng
                        came_from[nk] = node
                        f = ng + self.H_WEIGHT * heur(nxt[0], nxt[1])
                        heapq.heappush(open_heap, (f, counter, nxt, ng, d, steer))
                        counter += 1

        # ---- 경로 복원 ----
        self.waypoints = []
        if found is not None:
            chain = [found]
            key = self._disc(*found)
            while key in came_from:
                prev = came_from[key]
                chain.append(prev)
                key = self._disc(*prev)
            chain.reverse()
            # 각 점에 진행 방향(dir) 부여: 다음 점이 차 앞쪽이면 +1, 뒤쪽이면 -1
            for i, (x, y, yaw) in enumerate(chain):
                if i + 1 < len(chain):
                    nx, ny, _ = chain[i + 1]
                    ahead = math.cos(yaw) * (nx - x) + math.sin(yaw) * (ny - y)
                    d = 1.0 if ahead >= 0 else -1.0
                else:
                    d = self.waypoints[-1][3] if self.waypoints else 1.0
                self.waypoints.append((x, y, yaw, d))
            print(f"[plan] path found: {len(self.waypoints)} pts, pops={pops}")
        else:
            print(f"[plan] NO PATH (pops={pops}) - target may be unreachable")
        self._planned = True

    # =====================================================================
    # 3) 경로 추종 (참고용 - 제어 담당이 개선 예정)
    # =====================================================================
    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        st = obs["state"]
        limits = obs["limits"]
        x, y, yaw, v = float(st["x"]), float(st["y"]), float(st["yaw"]), float(st["v"])
        max_steer = float(limits["maxSteer"])

        # 첫 호출 때 경로를 한 번 계획
        if not self._planned:
            self.compute_path(obs)

        cmd = {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": "D"}
        if not self.waypoints:
            cmd["brake"] = 1.0     # 경로가 없으면 정지
            return cmd

        # 가장 가까운 경로점 찾기
        di = [(math.hypot(wx - x, wy - y), i) for i, (wx, wy, _, _) in enumerate(self.waypoints)]
        _, near = min(di)
        # 그 앞쪽으로 look-ahead 점 선택(약 1.5m 앞)
        Ld = 1.5
        tgt = self.waypoints[-1]
        for i in range(near, len(self.waypoints)):
            wx, wy, wyaw, wdir = self.waypoints[i]
            if math.hypot(wx - x, wy - y) >= Ld:
                tgt = self.waypoints[i]
                break
        tx, ty, tyaw, tdir = tgt

        # 종료 판정: 마지막 점에 충분히 가까우면 정지
        gx, gy, _, _ = self.waypoints[-1]
        if math.hypot(gx - x, gy - y) < 0.5 and near >= len(self.waypoints) - 3:
            cmd["brake"] = 1.0
            return cmd

        # 진행 방향(기어)
        cmd["gear"] = "D" if tdir >= 0 else "R"

        # 순수 추종(Pure Pursuit) 조향: 후진이면 부호 반전
        ang_to_tgt = math.atan2(ty - y, tx - x)
        if tdir >= 0:
            alpha = _wrap(ang_to_tgt - yaw)
        else:
            alpha = _wrap(ang_to_tgt - (yaw + math.pi))
        dist = max(0.5, math.hypot(tx - x, ty - y))
        steer = math.atan2(2.0 * self.WB * math.sin(alpha), dist)
        if tdir < 0:
            steer = -steer
        cmd["steer"] = max(-max_steer, min(max_steer, steer))

        # 속도: 주차이므로 천천히(약 1.2 m/s). 목표 근처면 더 느리게
        remain = math.hypot(gx - x, gy - y)
        target_speed = min(1.2, 0.4 * remain + 0.3)
        if abs(v) < target_speed:
            cmd["accel"] = 0.3
        else:
            cmd["brake"] = 0.15
        return cmd


planner = PlannerSkeleton()

STUDENT_REPLAY_DIR = "student_replays"

def _slugify(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    slug = slug.strip("_")
    return slug or "session"

def save_student_replay(frames: List[Dict[str, Any]], meta: Dict[str, Any]) -> Optional[str]:
    if not frames:
        return None
    try:
        os.makedirs(STUDENT_REPLAY_DIR, exist_ok=True)
    except Exception as exc:
        print(f"[algo] replay dir error: {exc}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_key = meta.get("map_key") or meta.get("map_name") or "session"
    filename = f"{timestamp}_{_slugify(map_key)}.json"
    path = os.path.join(STUDENT_REPLAY_DIR, filename)
    payload = {
        "meta": meta,
        "frames": frames,
    }
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"[algo] replay saved: {path}")
        return path
    except Exception as exc:
        print(f"[algo] replay save failed: {exc}")
        return None


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:

    try:
        return planner.compute_control(obs)
    except Exception as exc:
        print(f"[algo] planner_step error: {exc}")
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}


def run_session(sock: socket.socket, peer: Tuple[str, int]) -> None:
    """시뮬레이터와의 단일 TCP 세션을 처리."""

    print(f"[algo] connected to simulator at {peer}")
    buffer = b""
    frames: List[Dict[str, Any]] = []
    session_meta: Dict[str, Any] = {
        "peer": {"host": peer[0], "port": peer[1]},
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "map_key": None,
        "map_name": None,
    }

    try:
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                print("[algo] simulator closed the connection")
                break

            buffer += chunk

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue

                try:
                    packet = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    print(f"[algo] bad JSON from simulator: {exc}")
                    continue

                if isinstance(packet, dict) and "map" in packet:
                    planner.set_map(packet["map"])
                    print("[algo] received static map payload")
                    map_payload = packet["map"]
                    session_meta["map_key"] = map_payload.get("key")
                    session_meta["map_name"] = map_payload.get("name")
                    session_meta["map_extent"] = map_payload.get("extent")
                    session_meta["slots_total"] = len(map_payload.get("slots", []))
                    continue

                try:
                    cmd = planner_step(packet)
                    payload = json.dumps(cmd, ensure_ascii=False) + "\n"
                    sock.sendall(payload.encode("utf-8"))
                    frames.append({
                        "t": packet.get("t"),
                        "obs": packet,
                        "cmd": cmd,
                    })
                except BrokenPipeError:
                    print("[algo] send failed: broken pipe")
                    return
                except Exception as exc:
                    print(f"[algo] planner/send error: {exc}")

    except (ConnectionResetError, ConnectionAbortedError) as exc:
        print(f"[algo] connection error: {exc}")
    except Exception as exc:
        print(f"[algo] unexpected error while talking to simulator: {exc}")
    finally:
        session_meta["end_time"] = datetime.now().isoformat(timespec="seconds")
        session_meta["frame_count"] = len(frames)
        save_student_replay(frames, session_meta)


def run_client(host: str, port: int) -> None:
    """시뮬레이터가 열어둔 포트에 접속해 세션을 유지한다."""

    backoff = 1.0
    while True:
        try:
            print(f"[algo] connecting to simulator at {host}:{port} ...")
            with socket.create_connection((host, port), timeout=2.0) as sock:
                sock.settimeout(0.2)
                run_session(sock, sock.getpeername())
                backoff = 1.0  # 연결이 정상 종료되면 지연을 초기화
        except KeyboardInterrupt:
            print("\n[algo] stopping by keyboard interrupt")
            break
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            print(f"[algo] connect failed ({exc}); retrying in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff + 0.5, 5.0)
            continue

        # 시뮬레이터가 연결을 닫은 경우 짧게 대기 후 재시도
        print("[algo] lost connection - waiting 1.0s before retry")
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55556)
    options = parser.parse_args()

    # Ctrl+C 입력 시 즉시 종료
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    run_client(options.host, options.port)


if __name__ == "__main__":
    main()