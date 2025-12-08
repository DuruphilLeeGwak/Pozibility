import numpy as np
from enum import Enum
from typing import Tuple, Optional

# 상수 정의
LOWER_INDICES = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles


class BodyType(Enum):
    FULL = "full"
    UPPER = "upper"


class AlignmentCase(Enum):
    A = "A"  # Full -> Full
    B = "B"  # Upper -> Upper
    C = "C"  # Full -> Upper
    D = "D"  # Upper -> Full


class AlignManager:
    def __init__(self, config):
        self.config = config

    def determine_case(
        self,
        src_kpts,
        src_scores,
        ref_kpts,
        ref_scores,
        src_img_size=None,
        ref_img_size=None,
    ):
        """
        src/ref의 Body Type을 판별하고 케이스 결정
        """
        print("\n🔍 [DEBUG] AlignManager.determine_case()")

        src_type = self._get_type(src_scores, src_kpts, "src", src_img_size)
        ref_type = self._get_type(ref_scores, ref_kpts, "ref", ref_img_size)

        if src_type == BodyType.FULL and ref_type == BodyType.FULL:
            case = AlignmentCase.A
        elif src_type == BodyType.UPPER and ref_type == BodyType.UPPER:
            case = AlignmentCase.B
        elif src_type == BodyType.FULL and ref_type == BodyType.UPPER:
            case = AlignmentCase.C
        else:
            case = AlignmentCase.D

        print(f"   Result: {src_type.value} -> {ref_type.value} = Case {case.value}")

        return src_type, ref_type, case

    def _get_type(self, scores, kpts, label, img_size=None):
        """
        Body Type 판별 (v3.1: 해부학 + Ghost Leg 강화)

        전신(FULL) 판별 조건 (모두 충족해야 함):
        1. 하반신 키포인트 점수 >= lower_body_conf_threshold 개수 충족
        2. 해부학적 순서: hip.y < knee.y < ankle.y (양쪽 다리 모두)
        3. knee/ankle 점수가 ghost_threshold 이상 (무릎/발목 모두)

        하나라도 실패하면 상반신(UPPER)
        """
        print(f"\n   🔍 [DEBUG] _get_type({label})")

        # 설정값 가져오기
        kpt_threshold = getattr(self.config, "kpt_threshold", 0.1)
        min_valid = getattr(self.config, "full_body_min_valid_lower", 4)
        ghost_score_threshold = getattr(self.config, "ghost_score_threshold", 2.0)
        # 🔹 하반신 전용 confidence threshold (없으면 기본 2.0)
        lower_thr = getattr(self.config, "lower_body_confidence_threshold", 2.0)

        # ============================================================
        # [조건 1] 하반신 키포인트 점수 체크
        # ============================================================
        print(f"      [조건 1] 하반신 점수 체크 (threshold={lower_thr})")

        valid_count = 0
        for i in LOWER_INDICES:
            if i < len(scores):
                score = scores[i]
                is_valid = score > lower_thr
                if is_valid:
                    valid_count += 1

        score_check_pass = valid_count >= min_valid
        print(
            f"         valid_count: {valid_count} >= {min_valid}? "
            f"{'PASS ✅' if score_check_pass else 'FAIL ❌'}"
        )

        if not score_check_pass:
            print("      → UPPER (점수 체크 실패)")
            return BodyType.UPPER

        # ============================================================
        # [조건 2] 해부학적 순서 검증 (핵심!)
        # 정상 전신: hip.y < knee.y < ankle.y
        # Ghost Leg: 이 순서가 깨짐 (DWPose가 추측한 경우)
        # ============================================================
        print("\n      [조건 2] 해부학적 순서 검증 (hip.y < knee.y < ankle.y)")

        # 골반이 없는 경우도 처리
        l_hip_y = (
            kpts[11][1]
            if 11 < len(kpts) and scores[11] > kpt_threshold
            else None
        )
        r_hip_y = (
            kpts[12][1]
            if 12 < len(kpts) and scores[12] > kpt_threshold
            else None
        )
        l_knee_y = (
            kpts[13][1]
            if 13 < len(kpts) and scores[13] > kpt_threshold
            else None
        )
        r_knee_y = (
            kpts[14][1]
            if 14 < len(kpts) and scores[14] > kpt_threshold
            else None
        )
        l_ankle_y = (
            kpts[15][1]
            if 15 < len(kpts) and scores[15] > kpt_threshold
            else None
        )
        r_ankle_y = (
            kpts[16][1]
            if 16 < len(kpts) and scores[16] > kpt_threshold
            else None
        )

        def check_leg_order(hip_y, knee_y, ankle_y, side):
            """한쪽 다리의 해부학적 순서 검증"""
            if hip_y is None or knee_y is None:
                print(f"         {side}: hip 또는 knee 없음 → 검증 불가")
                return False

            # hip.y < knee.y 체크 (무릎이 골반보다 아래에 있어야 함)
            hip_knee_ok = hip_y < knee_y

            # ankle이 있으면 knee.y < ankle.y도 체크
            if ankle_y is not None:
                knee_ankle_ok = knee_y < ankle_y
                order_ok = hip_knee_ok and knee_ankle_ok
                print(
                    f"         {side}: hip.y({hip_y:.0f}) < "
                    f"knee.y({knee_y:.0f}) < ankle.y({ankle_y:.0f})?"
                )
                print(
                    f"                hip<knee: {hip_knee_ok}, "
                    f"knee<ankle: {knee_ankle_ok} → "
                    f"{'OK ✅' if order_ok else 'FAIL ❌'}"
                )
            else:
                order_ok = hip_knee_ok
                print(
                    f"         {side}: hip.y({hip_y:.0f}) < knee.y({knee_y:.0f})? "
                    f"{hip_knee_ok} → {'OK ✅' if order_ok else 'FAIL ❌'}"
                )

            return order_ok

        left_order_ok = check_leg_order(l_hip_y, l_knee_y, l_ankle_y, "Left ")
        right_order_ok = check_leg_order(r_hip_y, r_knee_y, r_ankle_y, "Right")

        # 🔹 양쪽 다리가 모두 정상일 때만 FULL 인정
        anatomy_check_pass = left_order_ok and right_order_ok
        print(f"         Both legs OK? {'PASS ✅' if anatomy_check_pass else 'FAIL ❌'}")

        if not anatomy_check_pass:
            print("      → UPPER (해부학적 순서 검증 실패 - Ghost Leg)")
            return BodyType.UPPER

        # ============================================================
        # [조건 3] Ghost Score 검증
        # DWPose 정상 점수: 3.0 ~ 8.0
        # 추측 점수: 0.5 ~ 2.0
        # knee/ankle 점수가 너무 낮으면 Ghost
        # ============================================================
        print(f"\n      [조건 3] Ghost Score 검증 (threshold={ghost_score_threshold})")

        knee_scores = []
        ankle_scores = []

        if 13 < len(scores):
            knee_scores.append(scores[13])
        if 14 < len(scores):
            knee_scores.append(scores[14])
        if 15 < len(scores):
            ankle_scores.append(scores[15])
        if 16 < len(scores):
            ankle_scores.append(scores[16])

        max_knee_score = max(knee_scores) if knee_scores else 0
        max_ankle_score = max(ankle_scores) if ankle_scores else 0

        print(
            f"         knee scores: {[f'{s:.2f}' for s in knee_scores]}, "
            f"max={max_knee_score:.2f}"
        )
        print(
            f"         ankle scores: {[f'{s:.2f}' for s in ankle_scores]}, "
            f"max={max_ankle_score:.2f}"
        )

        # 🔹 knee & ankle 둘 다 threshold 이상이어야 FULL
        ghost_check_pass = (
            max_knee_score >= ghost_score_threshold
            and max_ankle_score >= ghost_score_threshold
        )
        print(
            f"         max_knee({max_knee_score:.2f}) >= {ghost_score_threshold} "
            f"and max_ankle({max_ankle_score:.2f}) >= {ghost_score_threshold}? "
            f"{'PASS ✅' if ghost_check_pass else 'FAIL ❌'}"
        )

        if not ghost_check_pass:
            print("      → UPPER (Ghost Score 검증 실패 - 낮은 신뢰도)")
            return BodyType.UPPER

        # ============================================================
        # 모든 조건 통과 → FULL
        # ============================================================
        print("      → FULL (모든 조건 통과)")
        return BodyType.FULL

    def calc_scale(self, src_face_size, ref_face_size):
        if not self.config.face_scale_enabled or ref_face_size < 1:
            return 1.0
        scale = np.clip(src_face_size / ref_face_size, 0.3, 3.0)
        return 1.0 if abs(scale - 1.0) < 0.05 else scale

    def align_coordinates(
        self, kpts, scores, case, src_person_bbox, src_face_bbox, face_bbox_func
    ):
        """
        좌표(kpts) 자체를 이동(Shift)시켜 정렬
        """
        print("\n" + "=" * 60)
        print(f"🔍 [DEBUG] AlignManager.align_coordinates(Case {case.value})")
        print("=" * 60)

        aligned_kpts = kpts.copy()

        if case == AlignmentCase.A:
            print("\n🦶 Case A: Bottom-based alignment")

            src_bottom = src_person_bbox.bbox[3]
            print(f"   src_person_bbox: {src_person_bbox.bbox}")
            print(f"   src_bottom (y2): {src_bottom}")

            feet_idx = [15, 16, 17, 18, 19, 20, 21, 22]
            valid_y = []
            for i in feet_idx:
                if i < len(scores) and scores[i] > 0.1:
                    valid_y.append(kpts[i][1])

            trans_bottom = max(valid_y) if valid_y else 0
            print(f"   trans_bottom (max valid feet): {trans_bottom:.1f}")

            if trans_bottom > 0:
                shift_y = src_bottom - trans_bottom
                aligned_kpts[:, 1] += shift_y
                print(f"   ✅ shift_y = {shift_y:.1f}")
            else:
                print("   ❌ trans_bottom = 0, NO SHIFT")

        else:
            print(f"\n👤 Case {case.value}: Face-based alignment")

            src_cx, src_cy = src_face_bbox.center
            print(f"   src_face_center: ({src_cx:.1f}, {src_cy:.1f})")

            trans_face_info = face_bbox_func(kpts, scores)
            trans_cx, trans_cy = trans_face_info.center
            print(f"   trans_face_center: ({trans_cx:.1f}, {trans_cy:.1f})")

            shift_x = src_cx - trans_cx
            shift_y = src_cy - trans_cy

            aligned_kpts[:, 0] += shift_x
            aligned_kpts[:, 1] += shift_y
            print(f"   ✅ shift: ({shift_x:.1f}, {shift_y:.1f})")

        print("=" * 60)
        return aligned_kpts
