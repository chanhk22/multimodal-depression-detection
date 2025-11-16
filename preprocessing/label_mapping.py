import os, glob, pandas as pd, numpy as np

# default threshold for PHQ (PHQ-8): commonly 10 for moderate depression
DEFAULT_PHQ_THRESHOLD = 10.0

def _read_all_csvs(labels_dir):
    rows = []
    for p in glob.glob(os.path.join(labels_dir, "*.csv")):
        try:
            df = pd.read_csv(p)
            df['__source_file'] = os.path.basename(p)
            rows.append(df)
        except Exception as e:
            print(f"[label_mapping] failed to read {p}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)

def canonicalize_column_names(df, dataset_hint=None):
    # lowercase keys
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    # find participant id column
    pid = None
    pid_candidates = ['participant_id','participant','id', 'index', 'session']
    

    for cand in pid_candidates:
        if cand in cols:
            pid = cols[cand]; break
        
    # find PHQ score/binary
    phq_score = None
    phq_score_candidates = [
        'phq8_score', 'phq_score', 'phq8', 'phq_total', 'phq',
        'phq-8_score', 'phq-8', 'depression_score'
    ]
    for cand in phq_score_candidates:
        if cand in cols:
            phq_score = cols[cand]
            break
    
    # find PHQ binary column (multiple patterns)
    phq_binary = None
    phq_binary_candidates = [
        'phq8_binary', 'phq_binary',  
        'phq-8_binary', 'depression_binary',
        'depression', 'depressed'
    ]
    # D-VLOG uses 'label' for depression/normal
    if dataset_hint == "D-VLOG":
        phq_binary_candidates.insert(0, 'label')
    
    for cand in phq_binary_candidates:
        if cand in cols:
            phq_binary = cols[cand]
            break
    gender = None
    for cand in ['gender','sex']:
        if cand in cols:
            gender = cols[cand]; break

    # find fold/split information (for D-VLOG)
    fold = None
    if dataset_hint == "D-VLOG":
        fold_candidates = ['fold', 'split', 'set']
        for cand in fold_candidates:
            if cand in cols:
                fold = cols[cand]
                break
    
    return {
        'pid': pid, 
        'phq_score': phq_score, 
        'phq_binary': phq_binary, 
        'gender': gender,
        'fold': fold
    }


def _process_binary_label(value, dataset_hint=None):
    """Convert various binary label formats to 0/1"""
    if pd.isna(value):
        return None
    
    value_str = str(value).lower().strip()
    
    # D-VLOG format: 'depression'/'normal'
    if dataset_hint == "D-VLOG":
        if value_str in ['depression', 'depressed', '1']:
            return 1
        elif value_str in ['normal', 'control', '0']:
            return 0
        else:
            return None
    
    # Standard binary formats
    if value_str in ['1', '1.0', 'depression', 'depressed']:
        return 1
    elif value_str in ['0', '0.0', 'normal', 'control']:
        return 0
    else:
        try:
            # Try to convert to number
            num_val = float(value)
            return 1 if num_val > 0.5 else 0
        except (ValueError, TypeError):
            return None
        
def load_labels(labels_dir_or_file, dataset_hint=None, phq_threshold=DEFAULT_PHQ_THRESHOLD):
    """
    Load labels from directory (DAIC/E-DAIC) or single file (D-VLOG)
    Returns mapping: session_id -> {'PHQ_Score': float or None, 'PHQ_Binary': 0/1 or None, 'Gender': str or None, 'Fold': str or None}
    """
    
    # Handle single file (D-VLOG) vs directory (DAIC/E-DAIC)
    if os.path.isfile(labels_dir_or_file):
        # Single CSV file (D-VLOG)
        try:
            df_all = pd.read_csv(labels_dir_or_file)
            df_all['__source_file'] = os.path.basename(labels_dir_or_file)
        except Exception as e:
            print(f"[label_mapping] failed to read {labels_dir_or_file}: {e}")
            return {}
    else:
        # Directory with multiple CSV files (DAIC/E-DAIC)
        df_all = _read_all_csvs(labels_dir_or_file)
        if df_all.empty:
            return {}

    mapping = {}
    cols = canonicalize_column_names(df_all, dataset_hint)
    pid_col = cols['pid']
    
    if pid_col is None:
        print(f"[label_mapping] Warning: No participant ID column found in {dataset_hint}")
        return {}
    
    
    print(f"[label_mapping] Processing {len(df_all)} rows for {dataset_hint}")
    print(f"[label_mapping] Found columns: {cols}")
    
    for _, r in df_all.iterrows():
        pid = r.get(pid_col)
        if pd.isna(pid):
            continue
        
        # Convert PID to string (handle numeric IDs)
        if isinstance(pid, (float, np.floating)) and float(pid).is_integer():
            sid = str(int(pid))
        else:
            sid = str(pid)
        
        entry = mapping.get(sid, {})
        
        # PHQ score
        if cols['phq_score'] and pd.notna(r.get(cols['phq_score'])):
            try:
                entry['PHQ_Score'] = float(r.get(cols['phq_score']))
            except (ValueError, TypeError):
                entry['PHQ_Score'] = None
        
        # PHQ binary (with dataset-specific processing)
        if cols['phq_binary'] and pd.notna(r.get(cols['phq_binary'])):
            binary_val = _process_binary_label(r.get(cols['phq_binary']), dataset_hint)
            if binary_val is not None:
                entry['PHQ_Binary'] = binary_val
        
        # Gender
        if cols['gender'] and pd.notna(r.get(cols['gender'])):
            gender_val = str(r.get(cols['gender'])).strip()
            # Normalize gender values
            gender_normalized = gender_val.lower()
            if gender_normalized in ['m', 'male', '1']:
                entry['Gender'] = 'Male'
            elif gender_normalized in ['f', 'female', '0']:
                entry['Gender'] = 'Female'
            else:
                entry['Gender'] = gender_val  # Keep original if unclear
        
        # Fold information (D-VLOG)
        if cols['fold'] and pd.notna(r.get(cols['fold'])):
            entry['Fold'] = str(r.get(cols['fold'])).strip()
        
        # raw row (last wins if duplicate sessions)
        entry['raw_row'] = r.to_dict()
        mapping[sid] = entry

    # postprocess: ensure binary exists (derive from score if missing)
    derived_count = 0
    for sid, e in mapping.items():
        if 'PHQ_Binary' not in e or e['PHQ_Binary'] is None:
            if 'PHQ_Score' in e and e['PHQ_Score'] is not None:
                e['PHQ_Binary'] = 1 if e['PHQ_Score'] >= phq_threshold else 0
                derived_count += 1
    
    if derived_count > 0:
        print(f"[label_mapping] Derived binary labels from scores for {derived_count} sessions")

    # consistency check: warn if mismatch between score and binary
    inconsistencies = []
    for sid, e in mapping.items():
        if ('PHQ_Score' in e and e['PHQ_Score'] is not None and 
            'PHQ_Binary' in e and e['PHQ_Binary'] is not None):
            derived = 1 if e['PHQ_Score'] >= phq_threshold else 0
            if derived != e['PHQ_Binary']:
                inconsistencies.append((sid, e['PHQ_Score'], e['PHQ_Binary'], derived))
    
    if inconsistencies:
        print(f"[label_mapping] Warning: found {len(inconsistencies)} PHQ score/binary inconsistencies:")
        for i, t in enumerate(inconsistencies[:5]):  # Show first 5
            print(f"  Session {t[0]}: score={t[1]}, binary={t[2]}, derived={t[3]}")
        if len(inconsistencies) > 5:
            print(f"  ... and {len(inconsistencies)-5} more")

    print(f"[label_mapping] Loaded {len(mapping)} sessions for {dataset_hint}")
    
    # Print summary statistics
    if mapping:
        phq_scores = [e.get('PHQ_Score') for e in mapping.values() if e.get('PHQ_Score') is not None]
        phq_binaries = [e.get('PHQ_Binary') for e in mapping.values() if e.get('PHQ_Binary') is not None]
        genders = [e.get('Gender') for e in mapping.values() if e.get('Gender') is not None]
        
        if phq_scores:
            print(f"[label_mapping] PHQ scores: mean={np.mean(phq_scores):.1f}, std={np.std(phq_scores):.1f}, range=[{min(phq_scores)}, {max(phq_scores)}]")
        
        if phq_binaries:
            depression_ratio = np.mean(phq_binaries)
            print(f"[label_mapping] Depression ratio: {depression_ratio:.3f} ({sum(phq_binaries)}/{len(phq_binaries)})")
        
        if genders:
            gender_counts = pd.Series(genders).value_counts()
            print(f"[label_mapping] Gender distribution: {gender_counts.to_dict()}")

    return mapping

# Test function
def test_label_loading():
    """Test function to verify label loading works correctly"""
    
    # Test DAIC-WOZ format
    print("=== Testing DAIC-WOZ format ===")
    # Example: labels directory with train/dev/test CSVs containing PHQ8_Score, PHQ8_Binary
    daic_labels = load_labels("/Users/2chanhyuk/Documents/data_raw/labels", dataset_hint="DAIC-WOZ")
    print(f"Loaded {daic_labels} DAIC-WOZ labels")
    # Test E-DAIC format  
    print("=== Testing E-DAIC format ===")
    # Example: labels directory with CSVs containing phq_score, phq_binary
    
    # Test D-VLOG format
    print("=== Testing D-VLOG format ===")
    # Example: single labels.csv with index, label (depression/normal), gender, fold
    

    print("Label mapping tests completed!")

if __name__ == "__main__":
    test_label_loading()