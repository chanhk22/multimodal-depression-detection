# audio_trim.py

import os
import yaml
import glob
import json
from preprocessing.daic_audio_pipeline import find_ellie_start, trim_wav_from_start

def trim_audio():
    with open("configs/default.yaml", encoding='utf-8') as f:
        C = yaml.safe_load(f)

    AUDIO_IN  = C['paths']['daic_woz']['audio_dir']
    TRS_IN    = C['paths']['daic_woz']['transcript_dir']
    PROC_ROOT = C['outputs']['processed_root']
    ELLIE_RGX = C['preprocessing']['ellie_regex']

    os.makedirs(os.path.join(PROC_ROOT, "Audio"), exist_ok=True)

    t0_values = {}  # t0 값을 저장할 딕셔너리

    # 트랜스크립트 파일들 처리
    for trs in sorted(glob.glob(os.path.join(TRS_IN, "*_TRANSCRIPT.csv"))):
        sid = os.path.basename(trs).split('_')[0]
        wav_in  = os.path.join(AUDIO_IN,  f"{sid}_AUDIO.wav")
        wav_out = os.path.join(PROC_ROOT, "Audio", f"{sid}_AUDIO_trimmed.wav")

        # 트리밍된 오디오 파일이 이미 존재하는지 확인
        if os.path.exists(wav_out): 
            print(f"Trimmed audio already exists for session {sid}: {wav_out}")
            continue  # 파일이 이미 존재하면 트리밍을 건너뜀

        if not os.path.exists(wav_in): 
            print(f"Audio file does not exist: {wav_in}")
            continue
        # Ellie start time
        try:
            t0 = find_ellie_start(trs, ELLIE_RGX)
            print(f"Found Ellie start time: {t0} for session {sid}")
            t0_values[sid] = t0  # t0 값을 딕셔너리에 저장
        except Exception as e:
            print(f"Error finding Ellie start time for session {sid}: {e}")
            continue

        # Audio trimming
        try:
            trim_wav_from_start(wav_in, wav_out, t0)
            print(f"Successfully trimmed audio for session {sid}")
        except Exception as e:
            print(f"Error trimming audio for session {sid}: {e}")
            continue

    # t0 값을 JSON 파일로 저장
    with open('t0_values.json', 'w') as f:
        json.dump(t0_values, f)

    print("DAIC head cut t0 filter done.")

if __name__ == "__main__":
    trim_audio()
