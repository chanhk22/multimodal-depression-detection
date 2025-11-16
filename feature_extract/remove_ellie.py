import os
import yaml
import glob
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def load_participant_segments(transcript_path, t0_start=0.0):
    """
    TRANSCRIPT 파일에서 Participant가 말한 시간 구간 목록을 추출합니다.
    t0_start 이후의 발화만 반환합니다.

    Args:
        transcript_path: TRANSCRIPT CSV 파일 경로
        t0_start: 시작 시간(초) — 이 시간 이전 발화는 무시
    Returns:
        List of tuples: [(start_time, stop_time), ...]
    """
    try:
        df = pd.read_csv(transcript_path, delimiter='\t')

        # filtering 'Participant' row only
        participant_df = df[df['speaker'] == 'Participant'].copy()

        # start_time, stop_time to float (robust 처리)
        participant_df['start_time'] = participant_df['start_time'].astype(float)
        participant_df['stop_time'] = participant_df['stop_time'].astype(float)

        # t0 이후의 발화만 남김
        participant_df = participant_df[participant_df['start_time'] >= t0_start]

        # (start_time, stop_time) tuple list
        segments = list(zip(participant_df['start_time'], participant_df['stop_time']))

        return segments

    except Exception as e:
        print(f"Error loading transcript {transcript_path}: {e}")
        return []


def filter_timestamp_feature(feature_path, segments, output_path):
    """
    Timestamp 컬럼이 있는 Feature 파일을 필터링합니다 (CLNF 계열).
    """
    try:
        df = pd.read_csv(feature_path, sep=',', skipinitialspace=True)

        # timestamp 컬럼 찾기 (공백 포함 가능성 고려)
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break

        if timestamp_col is None:
            print(f"Warning: No timestamp column found in {feature_path}")
            return

        # timestamp 컬럼을 float으로 변환 (문자열로 되어 있을 수 있음)
        try:
            df[timestamp_col] = df[timestamp_col].astype(float)
        except Exception:
            df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
            df = df.dropna(subset=[timestamp_col])

        # 각 segment에 해당하는 행들을 수집
        filtered_segments = []
        for start, stop in segments:
            mask = (df[timestamp_col] >= start) & (df[timestamp_col] <= stop)
            segment_df = df.loc[mask]
            if not segment_df.empty:
                filtered_segments.append(segment_df)

        # 모든 segment를 하나로 합치기
        if filtered_segments:
            result_df = pd.concat(filtered_segments, ignore_index=True)

            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # CSV로 저장
            result_df.to_csv(output_path, index=False)
            print(f"✓ Saved: {output_path} ({len(result_df)} rows)")
        else:
            print(f"Warning: No data found for {feature_path}")

    except Exception as e:
        print(f"Error processing {feature_path}: {e}")


def filter_row_based_feature(feature_path, segments, output_path, row_duration=0.01):
    """
    Row 기반 Feature 파일을 필터링합니다 (COVAREP, Formant).
    row_duration: 한 행당 시간 (초 단위, 기본값 0.01s)
    """
    try:
        df = pd.read_csv(feature_path)

        filtered_segments = []
        for start, stop in segments:
            # 행 인덱스 범위를 계산
            start_row = int(start / row_duration)
            # stop_row를 +1 하여 종료시간에 해당하는 행을 포함하도록 함
            stop_row = int(stop / row_duration) + 1

            # 범위 체크 및 슬라이싱 (iloc의 end는 exclusive)
            if start_row < len(df):
                end_row = min(stop_row, len(df))
                segment_df = df.iloc[start_row:end_row]
                if not segment_df.empty:
                    filtered_segments.append(segment_df)

        if filtered_segments:
            result_df = pd.concat(filtered_segments, ignore_index=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
            print(f"✓ Saved: {output_path} ({len(result_df)} rows)")
        else:
            print(f"Warning: No data found for {feature_path}")

    except Exception as e:
        print(f"Error processing {feature_path}: {e}")


def filter_features():
    """
    메인 함수: 모든 Feature 파일을 세션별로 필터링합니다.
    """
    # Config 파일 로드
    with open("configs/default.yaml", encoding='utf-8') as f:
        C = yaml.safe_load(f)

    FEAT_IN = C['paths']['daic_woz']['features_dir']
    PROC_ROOT = C['outputs']['processed_root']
    TRS_IN = C['paths']['daic_woz']['transcript_dir']

    # t0_values.json 위치를 스크립트 기준으로 찾음 (프로젝트 루트 가정)
    script_path = Path(__file__).resolve()            # .../feature_extract/remove_ellie.py
    project_root = script_path.parent.parent           # 부모의 부모 -> 프로젝트 루트
    t0_file = project_root / "t0_values.json"

    print(f"(debug) script_path: {script_path}")
    print(f"(debug) project_root: {project_root}")
    print(f"(debug) looking for t0_file at: {t0_file}")

    t0_dict = {}

    if t0_file.exists():
        try:
            with t0_file.open('r', encoding='utf-8') as f:
                raw = json.load(f)

            # Normalize keys to strings and values to floats where possible
            for k, v in raw.items():
                try:
                    t0_dict[str(k)] = float(v)
                except Exception:
                    print(f"Warning: ignored non-numeric t0 for key {k!r}: {v!r}")

            if t0_dict:
                print(f"✅ Loaded {len(t0_dict)} t0 entries from {t0_file}")
                print("Sample t0 entries:", list(t0_dict.items())[:10])
            else:
                print(f"⚠ {t0_file} parsed but contains no usable numeric entries.")
        except Exception as e:
            print(f"Error reading {t0_file}: {e}")
            t0_dict = {}
    else:
        print(f"⚠ t0 file not found at {t0_file}. Using t0=0.0 for all sessions.")

    # Feature 유형별 설정
    timestamp_features = [
        "clnf_au",
        "clnf_feature",
        "clnf_feature3d",
        "clnf_gaze",
        "clnf_pose"
    ]
    row_based_features = [
        "covarep",
        "formant"
    ]

    # 모든 TRANSCRIPT 파일 찾기
    transcript_files = sorted(glob.glob(os.path.join(TRS_IN, "*_TRANSCRIPT.csv")))

    print(f"Found {len(transcript_files)} transcript files")
    print("=" * 60)

    # 각 세션 처리
    for transcript_path in tqdm(transcript_files, desc="Processing sessions"):
        sid = os.path.basename(transcript_path).split('_')[0]
        print(f"\n[Session {sid}]")

        # t0 가져오기: sid 키가 여러 포맷일 수 있으니 유연하게 체크
        t0 = t0_dict.get(str(sid),
                         t0_dict.get(f"{sid}_P",
                                    t0_dict.get(f"{sid}_participant", 0.0)))
        print(f"  (debug) Using t0 for session {sid!s}: {t0}")

        # Participant 발화 구간 로드 — t0를 전달
        segments = load_participant_segments(transcript_path, t0_start=t0)

        if not segments:
            print(f"  ⚠ No participant segments found (after t0={t0}), skipping session {sid}")
            continue

        print(f"  Found {len(segments)} participant segments (after t0={t0})")

        # 유형 1: Timestamp 기반 Feature 처리
        for feature_subdir in timestamp_features:
            feature_dir = os.path.join(FEAT_IN, feature_subdir)

            if not os.path.exists(feature_dir):
                print(f"  ⚠ Feature directory not found: {feature_dir}")
                continue

            feature_files = glob.glob(os.path.join(feature_dir, f"{sid}_*.txt"))

            for feature_path in feature_files:
                filename = os.path.basename(feature_path)
                output_path = os.path.join(
                    PROC_ROOT, "Features",
                    feature_subdir, filename.replace('.txt', '.csv')
                )

                if os.path.exists(output_path):
                    print(f"  ⊘ Already exists: {filename}")
                    continue

                filter_timestamp_feature(feature_path, segments, output_path)

        # 유형 2: Row 기반 Feature 처리
        for feature_subdir in row_based_features:
            feature_dir = os.path.join(FEAT_IN, feature_subdir)

            if not os.path.exists(feature_dir):
                print(f"  ⚠ Feature directory not found: {feature_dir}")
                continue

            feature_files = glob.glob(os.path.join(feature_dir, f"{sid}_*.csv"))

            for feature_path in feature_files:
                filename = os.path.basename(feature_path)
                output_path = os.path.join(
                    PROC_ROOT, "Features",
                    feature_subdir, filename
                )

                if os.path.exists(output_path):
                    print(f"  ⊘ Already exists: {filename}")
                    continue

                filter_row_based_feature(feature_path, segments, output_path)

    print("\n" + "=" * 60)
    print("✅ Feature filtering completed!")


if __name__ == "__main__":
    filter_features()
