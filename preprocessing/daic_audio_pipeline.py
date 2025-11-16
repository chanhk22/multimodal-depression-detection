# preprocessing/daic_audio_pipeline.py
import wave, contextlib
import pandas as pd

ELLIE_REGEX_DEFAULT = r"(?i)hi\s*i[' ]?m\s*ellie"

def find_ellie_start(transcript_csv, ellie_regex=ELLIE_REGEX_DEFAULT):
    try:
        df = pd.read_csv(transcript_csv, delimiter='\t')
        
        # 1. 파일이 비어있는지 먼저 확인
        if df.empty:
            print(f"Warning: Empty file {transcript_csv}. Returning 0.0")
            return 0.0
            
        # 2. 'value' 또는 'start_time' 컬럼이 없는지 확인
        if 'value' not in df.columns or 'start_time' not in df.columns:
            print(f"Warning: Missing columns in {transcript_csv}. Returning 0.0")
            return 0.0

        # 3. Ellie 정규식 매칭 시도
        m = df['value'].astype(str).str.contains(ellie_regex, na=False)
        
        # 4. ★★★ 로직 수정 ★★★
        if m.any():
            # (의도 1) Ellie를 찾았으면 -> 매칭된 첫 번째 행의 start_time 반환
            return float(df.loc[m, 'start_time'].iloc[0])
        else:
            # (의도 2) Ellie를 못 찾았으면 -> 파일의 맨 첫 번째 줄 start_time 반환
            print(f"Warning: Ellie regex not found in {transcript_csv}. Using file's first start_time as fallback.")
            return float(df['start_time'].iloc[0])
            
    except pd.errors.EmptyDataError:
        print(f"Error: EmptyDataError for {transcript_csv}. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"Error processing {transcript_csv}: {e}. Returning 0.0")
        return 0.0


def trim_wav_from_start(in_wav, out_wav, start_sec):
    if start_sec <= 0:
        # 그냥 복사
        import shutil; shutil.copyfile(in_wav, out_wav); return
    with contextlib.closing(wave.open(in_wav, 'rb')) as w:
        fr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
        n  = w.getnframes()
        start_frame = int(start_sec * fr)
        start_frame = max(0, min(start_frame, n))
        w.setpos(start_frame)
        frames = w.readframes(n - start_frame)

    with contextlib.closing(wave.open(out_wav, 'wb')) as wout:
        wout.setnchannels(ch)
        wout.setsampwidth(sw)
        wout.setnchannels(ch)
        wout.setframerate(fr)
        wout.writeframes(frames)

def process_session(audio_wav, transcript_csv, out_wav, ellie_regex=ELLIE_REGEX_DEFAULT):
    t0 = find_ellie_start(transcript_csv, ellie_regex=ellie_regex)
    trim_wav_from_start(audio_wav, out_wav, t0)
    return t0