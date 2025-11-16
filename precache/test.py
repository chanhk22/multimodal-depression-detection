# diagnose_session.py
import os, sys, glob, json
import pandas as pd
import numpy as np

# adjust these to match your config if needed
CONFIG_PATH = "configs/default.yaml"

def load_config(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_t0_dict():
    p = "t0_values.json"
    if os.path.exists(p):
        try:
            with open(p,'r',encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("Failed to read t0_values.json:", e)
            return {}
    return {}

def find_transcript(session, transcripts_dir):
    candidates = sorted(glob.glob(os.path.join(transcripts_dir, f"{session}_*TRANSCRIPT.csv")))
    return candidates[0] if candidates else None

def load_segments(transcript_path, t0=0.0):
    try:
        df = pd.read_csv(transcript_path, delimiter='\t')
        p = df[df['speaker'] == 'Participant'].copy()
        p['start_time'] = p['start_time'].astype(float)
        p['stop_time'] = p['stop_time'].astype(float)
        p = p[p['start_time'] >= t0]
        segs = list(zip(p['start_time'], p['stop_time']))
        return segs, p
    except Exception as e:
        return None, f"ERR:{e}"

def audio_files_info(session, proc_root):
    out = {}
    for sub,name_pattern in [('covarep', f"{session}_*.csv"), ('formant', f"{session}_*.csv")]:
        p = os.path.join(proc_root, "Features", sub)
        files = sorted(glob.glob(os.path.join(p, name_pattern)))
        out[sub] = {'files': files}
        if files:
            try:
                df = pd.read_csv(files[0])
                out[sub]['rows'] = len(df)
                out[sub]['cols'] = df.shape[1]
                out[sub]['sample_row0'] = df.iloc[0].to_dict()
            except Exception as e:
                out[sub]['error'] = str(e)
    return out

def visual_files_info(session, proc_root):
    clnfs = ["clnf_au","clnf_feature","clnf_feature3d","clnf_gaze","clnf_pose"]
    out = {}
    for sub in clnfs:
        p = os.path.join(proc_root, "Features", sub)
        files = sorted(glob.glob(os.path.join(p, f"{session}_*.csv")))
        out[sub] = {'files': files}
        if files:
            try:
                df = pd.read_csv(files[0], skipinitialspace=True)
                ts_col = None
                for c in df.columns:
                    if 'timestamp' in c.lower():
                        ts_col = c; break
                out[sub]['ts_col'] = ts_col
                if ts_col:
                    out[sub]['ts_head'] = df[ts_col].values[:5].tolist()
                    out[sub]['ts_tail'] = df[ts_col].values[-5:].tolist()
                    out[sub]['rows'] = len(df)
                    out[sub]['cols'] = df.shape[1]
            except Exception as e:
                out[sub]['error'] = str(e)
    return out

def simulate_windows(segments, audio_rows, audio_row_duration, visual_ts, window_sec=2.0, stride=1.0, n=10):
    # simulate first n windows and count frames
    if not segments:
        return []
    total_start = segments[0][0]
    total_end = segments[-1][1]
    t = total_start
    out=[]
    while t + window_sec <= total_end + 1e-8 and len(out) < n:
        ws, we = t, t + window_sec
        a_count = None
        v_count = None
        if audio_rows is not None:
            # audio timestamps built as np.arange(rows)*row_dur
            audio_ts = np.arange(audio_rows) * audio_row_duration
            mask_a = (audio_ts >= ws-1e-8) & (audio_ts <= we+1e-8)
            a_count = int(mask_a.sum())
        if visual_ts is not None and len(visual_ts)>0:
            mask_v = (visual_ts >= ws-1e-8) & (visual_ts <= we+1e-8)
            v_count = int(mask_v.sum())
        out.append({'window':[ws,we],'audio_frames':a_count,'visual_frames':v_count})
        t += stride
    return out

def main(argv):
    if len(argv)<2:
        print("Usage: python diagnose_session.py 300 [301 ...]")
        return
    config = load_config(CONFIG_PATH)
    proc_root = config['outputs']['processed_root']
    transcripts_dir = config['paths']['daic_woz']['transcript_dir']
    t0_dict = load_t0_dict()
    sessions = argv[1:]
    for s in sessions:
        print("\n"+"="*40)
        print("SESSION:", s)
        t0 = t0_dict.get(s, 0.0)
        print("t0:", t0)
        tr = find_transcript(s, transcripts_dir)
        print("transcript:", tr)
        if not tr:
            print("-> No transcript file found. That explains no windows.")
            continue
        segs, pd_or_err = load_segments(tr, t0)
        if segs is None:
            print("Error reading transcript:", pd_or_err)
            continue
        print("participant segments count:", len(segs))
        if len(segs)==0:
            print("-> No segments after applying t0. That explains no windows.")
            continue
        # audio info
        audio_info = audio_files_info(s, proc_root)
        print("Audio feature info:")
        for k,v in audio_info.items():
            print(" ",k,":",v)
        # visual info
        visual_info = visual_files_info(s, proc_root)
        print("Visual feature info:")
        for k,v in visual_info.items():
            print(" ",k,":",v)
        # derive audio_rows if present (choose covarep or formant)
        audio_rows = None
        if audio_info.get('covarep',{}).get('rows'):
            audio_rows = audio_info['covarep']['rows']
        elif audio_info.get('formant',{}).get('rows'):
            audio_rows = audio_info['formant']['rows']
        # visual timestamps choose first available
        visual_ts = None
        for k,v in visual_info.items():
            if v.get('rows'):
                try:
                    # read the timestamp column
                    f = v['files'][0]
                    df = pd.read_csv(f, skipinitialspace=True)
                    for c in df.columns:
                        if 'timestamp' in c.lower():
                            visual_ts = df[c].astype(float).values
                            break
                    if visual_ts is not None:
                        break
                except Exception as e:
                    pass
        print("Derived audio_rows:", audio_rows)
        print("Derived visual_ts length:", None if visual_ts is None else len(visual_ts))
        # simulate windows
        sim = simulate_windows(segs, audio_rows, 0.01, visual_ts, window_sec=2.0, stride=1.0, n=10)
        print("Simulated first windows (audio_frames, visual_frames):")
        for i,x in enumerate(sim):
            print(" ",i, x)
        print("="*40)

if __name__ == "__main__":
    main(sys.argv)
