"""헤드리스 평가 하니스 — GUI 없이 시뮬을 그대로 돌려 플래너를 채점/검증한다.

시뮬(`demo_self_parking_sim.py`)의 맵 로딩·동역학·충돌판정·성공게이트를 그대로 재사용해,
각 맵/시드별로 success / collision / timeout 과 최종 IoU, steering reversals, gear switches 를 집계한다.
경로계획 수정 후 회귀 확인용. (시뮬 GUI 를 띄우지 않으므로 빠르고 반복 가능.)

실행:
  cd self-parking-sim
  PYTHONPATH="$(pwd):$(pwd)/../self-parking-user-algorithms" ./.venv/bin/python ../self-parking-user-algorithms/headless_eval.py
"""
import os, math, sys, copy, random
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # GUI 없이 import 가능하게
import numpy as np
import demo_self_parking_sim as S
import student_planner as SP


def prepare(cfg, seed):
    """ensure_map_loaded 와 동일하게 맵을 준비하고 (M, payload, target_slot) 반환."""
    base = S.get_base_map(cfg["filename"])
    variant = cfg.get("variant", "")
    M = S.apply_map_variant(base, variant, seed) if variant else copy.deepcopy(base)
    S.resize_slots_to_vehicle(M)
    S.open_top_parking_lane(M)
    M.line_rects = S.compute_line_rects(M)
    payload = S.build_map_payload(M)
    payload["expected_orientation"] = cfg.get("expected_orientation")
    trng = random.Random((seed if seed is not None else 0) ^ 0x5F3759DF)
    free = np.where(~M.occupied_idx)[0]
    if free.size == 0:
        return None
    target_slot = tuple(M.slots[int(trng.choice(free.tolist()))].tolist())
    return M, payload, target_slot


def run_episode(cfg, seed):
    prep = prepare(cfg, seed)
    if prep is None:
        return {"why": "no_free"}
    M, payload, target_slot = prep
    P = S.Params()
    xmin, xmax, ymin, ymax = M.extent
    st = S.State(xmin + 4.0, ymin + 6.0, math.radians(90.0), 0.0)
    planner = SP.PlannerSkeleton()
    planner.set_map(payload)

    delta = 0.0; t = 0.0
    flips = 0; prev_sign = 0; gear_switches = 0; prev_gear = "D"
    last_iou = 0.0
    for _ in range(int(P.timeout / P.dt)):
        cmd = planner.compute_control(S.build_obs_payload(t, st, target_slot, P))
        steer = float(cmd.get("steer", 0.0)); accel = float(cmd.get("accel", 0.0))
        brake = float(cmd.get("brake", 0.0)); gear = cmd.get("gear", "D")
        # --- 시뮬 동역학 복제 (step_kinematic) ---
        delta = S.move_toward(delta, steer, P.steerRate * P.dt)
        if gear != prev_gear:
            gear_switches += 1; prev_gear = gear
        ssign = (1 if delta > 0 else -1) if abs(delta) >= S.STEER_FLIP_DEADZONE else 0
        if ssign != 0:
            if prev_sign != 0 and ssign != prev_sign:
                flips += 1
            prev_sign = ssign
        sgn = 1.0 if gear == "D" else -1.0
        vsgn = math.copysign(1.0, st.v) if abs(st.v) > 1e-6 else 0.0
        a = sgn * P.maxAccel * accel - vsgn * P.maxBrake * brake
        if accel < 1e-3 and brake < 1e-3 and abs(st.v) > 1e-3:
            a -= vsgn * P.coastDecel
        st = S.step_kinematic(st, delta, a, P); t += P.dt
        # --- 충돌/성공 판정 (시뮬과 동일) ---
        car = S.car_polygon(st, P)
        iou = S.compute_slot_iou(car, target_slot); last_iou = iou
        ori = S.determine_parking_orientation(st, target_slot)
        collided = False; reason = None
        if S.ENABLE_BOUNDARY_COLLISIONS:
            for vx, vy in car:
                if not (xmin <= vx <= xmax and ymin <= vy <= ymax):
                    collided, reason = True, "boundary"; break
        if not collided:
            for i, r in enumerate(M.slots):
                if M.occupied_idx[i] and S.poly_intersects_rect(car, tuple(r)):
                    collided, reason = True, "occupied_slot"; break
        if not collided:
            for r in M.line_rects:
                if S.poly_intersects_rect(car, tuple(r)):
                    collided, reason = True, "line"; break
        if S.rect_contains_poly(target_slot, car):
            collided, reason = False, None
        if collided:
            return {"why": "collision", "reason": reason, "t": round(t, 1),
                    "iou": round(iou, 3), "flips": flips, "gears": gear_switches}
        if iou >= S.PARKING_SUCCESS_IOU and abs(st.v) <= 0.2 and ori != "unknown":
            return {"why": "success", "t": round(t, 1), "iou": round(iou, 3),
                    "ori": ori, "flips": flips, "gears": gear_switches}
    return {"why": "timeout", "iou": round(last_iou, 3), "flips": flips, "gears": gear_switches}


def main(n_seeds=20):
    from collections import Counter
    for cfg in S.AVAILABLE_MAPS:
        outs = [run_episode(cfg, s) for s in range(n_seeds)]
        c = Counter(o["why"] for o in outs)
        succ = [o for o in outs if o["why"] == "success"]
        mi = sum(o["iou"] for o in succ) / len(succ) if succ else 0.0
        mt = sum(o["t"] for o in succ) / len(succ) if succ else 0.0
        mf = sum(o["flips"] for o in succ) / len(succ) if succ else 0.0
        mg = sum(o["gears"] for o in succ) / len(succ) if succ else 0.0
        print(f"{cfg['key']:16} ({cfg.get('expected_orientation')}): {dict(c)}  "
              f"| success: iou={mi:.3f} t={mt:.1f}s reversals={mf:.1f} gears={mg:.1f}")
        for s, o in enumerate(outs):
            if o["why"] != "success":
                print(f"      seed{s}: {o}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
