import os
import glob
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class UnifiedWindowCacheBuilder:
    """
    Unified window cache builder with GUARANTEED fixed frame counts.
    
    Key Design Principles:
    1. Frame-based windowing (not time-based masking) - guarantees exact frame counts
    2. Speech-time remapping - continuous participant-only timeline
    3. Train-only PCA fitting - no data leakage
    4. Efficient storage - npz format, no CSV for raw features
    """
    
    def __init__(self, config, window_duration=2.0, overlap_ratio=0.5, 
                 min_frames_ratio=0.9, pca_components={'audio': 50, 'visual': 50}):
        """
        Args:
            config: Configuration dictionary
            window_duration: Window length in seconds
            overlap_ratio: Overlap ratio (0.5 = 50% overlap)
            min_frames_ratio: Minimum valid frames ratio (0.9 = 90% of expected)
            pca_components: PCA components per modality
        """
        self.config = config
        self.window_duration = window_duration
        self.overlap_ratio = overlap_ratio
        self.min_frames_ratio = min_frames_ratio
        self.pca_components = pca_components
        
        # Feature sampling rates (Hz)
        self.audio_hz = 100.0  # 10ms = 0.01s per frame
        self.visual_hz = 30.0  # ~33ms per frame
        
        # Expected frames per window (EXACT values)
        self.expected_audio_frames = int(np.round(window_duration * self.audio_hz))
        self.expected_visual_frames = int(np.round(window_duration * self.visual_hz))
        
        # Stride in frames (not seconds!)
        self.audio_stride_frames = int(np.round(window_duration * (1 - overlap_ratio) * self.audio_hz))
        self.visual_stride_frames = int(np.round(window_duration * (1 - overlap_ratio) * self.visual_hz))
        
        # Minimum valid frames
        self.min_audio_frames = int(self.expected_audio_frames * min_frames_ratio)
        self.min_visual_frames = int(self.expected_visual_frames * min_frames_ratio)
        
        # Load t0 and labels
        self.t0_dict = self._load_t0_values()
        self.label_mappings = self._load_all_labels()
        
        print(f"\n{'='*60}")
        print(f"Window Configuration")
        print(f"{'='*60}")
        print(f"Duration: {window_duration}s")
        print(f"Overlap: {overlap_ratio*100}%")
        print(f"Audio: {self.expected_audio_frames} frames @ {self.audio_hz}Hz")
        print(f"  → stride: {self.audio_stride_frames} frames")
        print(f"  → min valid: {self.min_audio_frames} frames")
        print(f"Visual: {self.expected_visual_frames} frames @ {self.visual_hz}Hz")
        print(f"  → stride: {self.visual_stride_frames} frames")
        print(f"  → min valid: {self.min_visual_frames} frames")
        print(f"PCA components: {pca_components}")
        print(f"{'='*60}\n")
    
    def _load_t0_values(self):
        """Load t0 values from JSON"""
        t0_path = 't0_values.json'
        if os.path.exists(t0_path):
            with open(t0_path, 'r') as f:
                t0_dict = json.load(f)
            print(f"[t0] Loaded {len(t0_dict)} t0 values")
            return t0_dict
        else:
            print(f"[t0] Warning: not found, using 0.0 for all")
            return {}
    
    def _load_all_labels(self):
        """Load labels with PHQ_Score/PHQ8_Score fallback"""
        mappings = {}
        
        labels_path = self.config.get('paths', {}).get('daic_woz', {}).get('labels_dir')
        if not labels_path:
            print("[labels] Warning: No label path in config")
            mappings['DAIC-WOZ'] = {}
            return mappings
        
        labels_path = str(labels_path)
        print(f"[labels] Loading from: {labels_path}")
        
        split_files = [
            'train_split_Depression_AVEC2017.csv',
            'dev_split_Depression_AVEC2017.csv',
            'full_test_split.csv'
        ]
        
        merged_labels = {}
        
        for split_file in split_files:
            file_path = os.path.join(labels_path, split_file)
            if not os.path.exists(file_path):
                continue
            
            try:
                df = pd.read_csv(file_path)
                split_name = 'train' if 'train' in split_file.lower() else \
                            'dev' if 'dev' in split_file.lower() else 'test'
                
                print(f"  {split_file}: {len(df)} entries")
                
                for _, row in df.iterrows():
                    # Extract ID
                    pid = None
                    for col in ['Participant_ID', 'participant_id', 'ID', 'id']:
                        if col in df.columns and pd.notna(row[col]):
                            pid = str(int(row[col]))
                            break
                    
                    if pid is None:
                        continue
                    
                    # Priority: PHQ_Score > PHQ8_Score
                    phq_score = None
                    for col in ['PHQ_Score', 'PHQ8_Score']:
                        if col in df.columns and pd.notna(row[col]):
                            phq_score = float(row[col])
                            break
                    
                    # Priority: PHQ_Binary > PHQ8_Binary
                    phq_binary = None
                    for col in ['PHQ_Binary', 'PHQ8_Binary']:
                        if col in df.columns and pd.notna(row[col]):
                            phq_binary = float(row[col])
                            break
                    
                    # Derive if missing
                    if phq_binary is None and phq_score is not None:
                        phq_binary = 1.0 if phq_score >= 10.0 else 0.0
                    
                    gender = None
                    for col in ['Gender', 'gender']:
                        if col in df.columns and pd.notna(row[col]):
                            gender = str(row[col])
                            break
                    
                    label_dict = {
                        'PHQ_Score': phq_score,
                        'PHQ_Binary': phq_binary,
                        'Gender': gender,
                        'Fold': split_name
                    }
                    
                    # Multiple key formats
                    for key in [pid, f"{int(pid):03d}"]:
                        merged_labels[key] = label_dict
            
            except Exception as e:
                print(f"  Error: {e}")
        
        mappings['DAIC-WOZ'] = merged_labels
        
        # Statistics
        unique_ids = set(int(k) for k in merged_labels.keys() if k.isdigit())
        valid_count = sum(1 for k in unique_ids if merged_labels.get(str(k), {}).get('PHQ_Binary') is not None)
        
        print(f"[labels] Unique participants: {len(unique_ids)}")
        print(f"[labels] Valid labels: {valid_count}/{len(unique_ids)}")
        
        return mappings
    
    def _get_session_labels(self, session_id, dataset_name):
        """Get labels with fallback"""
        if dataset_name not in self.label_mappings:
            return None, None, None, None
        
        labels_dict = self.label_mappings[dataset_name]
        
        # Try different formats
        for key in [str(session_id), f"{int(session_id):03d}"]:
            if key in labels_dict:
                labels = labels_dict[key]
                return (labels.get('PHQ_Score'), 
                       labels.get('PHQ_Binary'),
                       labels.get('Gender'),
                       labels.get('Fold'))
        
        return None, None, None, None
    
    def _load_participant_segments(self, transcript_path, t0=0.0):
        """Load participant speech segments"""
        try:
            df = pd.read_csv(transcript_path, delimiter='\t')
            p_df = df[df['speaker'] == 'Participant'].copy()
            p_df['start_time'] = p_df['start_time'].astype(float)
            p_df['stop_time'] = p_df['stop_time'].astype(float)
            p_df = p_df[p_df['start_time'] >= t0]
            
            return list(zip(p_df['start_time'], p_df['stop_time']))
        except Exception as e:
            print(f"    Transcript error: {e}")
            return []
    
    def _remap_to_speech_frames(self, absolute_timestamps, features, segments):
        """
        Remap features to continuous speech-time FRAME indices.
        
        Returns:
            speech_features: (n_speech_frames, n_features) - continuous array
        """
        if absolute_timestamps is None or features is None or not segments:
            return None
        
        # Collect frame indices within segments
        selected_indices = []
        for start, stop in segments:
            mask = (absolute_timestamps >= start) & (absolute_timestamps <= stop)
            indices = np.where(mask)[0]
            selected_indices.extend(indices)
        
        if not selected_indices:
            return None
        
        # Extract frames in order - now a continuous array
        speech_features = features[selected_indices]
        
        return speech_features
    
    def _load_audio_features(self, session_id, proc_root, segments):
        """Load and remap audio to speech-frames"""
        features_list = []
        
        # COVAREP
        covarep_path = os.path.join(proc_root, "Features", "covarep", f"{session_id}_COVAREP.csv")
        if os.path.exists(covarep_path):
            try:
                df = pd.read_csv(covarep_path, on_bad_lines='skip', engine='python')
                if len(df) > 0:
                    features_list.append(df.values)
            except Exception as e:
                pass
        
        # Formant
        formant_path = os.path.join(proc_root, "Features", "formant", f"{session_id}_FORMANT.csv")
        if os.path.exists(formant_path):
            try:
                df = pd.read_csv(formant_path, on_bad_lines='skip', engine='python')
                if len(df) > 0:
                    features_list.append(df.values)
            except Exception as e:
                pass
        
        if not features_list:
            return None
        
        # Concatenate
        audio_features = np.hstack(features_list)
        
        # Generate absolute timestamps (0.01s per row)
        absolute_timestamps = np.arange(len(audio_features)) * 0.01
        
        # Remap to speech-frames
        speech_features = self._remap_to_speech_frames(absolute_timestamps, audio_features, segments)
        
        return speech_features
    
    def _load_visual_features(self, session_id, proc_root, segments):
        """Load and remap visual to speech-frames"""
        features_list = []
        absolute_timestamps = None
        
        clnf_types = ["clnf_au", "clnf_feature", "clnf_feature3d", "clnf_gaze", "clnf_pose"]
        
        for clnf_type in clnf_types:
            pattern = os.path.join(proc_root, "Features", clnf_type, f"{session_id}_*.csv")
            files = glob.glob(pattern)
            
            for file_path in files:
                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
                    if len(df) == 0:
                        continue
                    
                    # Find timestamp
                    timestamp_col = None
                    for col in df.columns:
                        if 'timestamp' in col.lower():
                            timestamp_col = col
                            break
                    
                    if timestamp_col is None:
                        continue
                    
                    if absolute_timestamps is None:
                        absolute_timestamps = df[timestamp_col].values
                    
                    # Numeric features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if timestamp_col in numeric_cols:
                        numeric_cols.remove(timestamp_col)
                    
                    if numeric_cols:
                        features_list.append(df[numeric_cols].values)
                
                except Exception as e:
                    pass
        
        if not features_list or absolute_timestamps is None:
            return None
        
        visual_features = np.hstack(features_list)
        
        # Remap to speech-frames
        speech_features = self._remap_to_speech_frames(absolute_timestamps, visual_features, segments)
        
        return speech_features
    
    def _extract_fixed_window(self, features, start_frame, expected_frames):
        """
        Extract EXACTLY expected_frames from features starting at start_frame.
        
        Strategy:
        - If enough frames available: extract exact count
        - If not enough: return None (window invalid)
        
        Returns:
            (frames, valid): frames array and validity flag
        """
        if features is None or start_frame < 0:
            return None, False
        
        end_frame = start_frame + expected_frames
        
        # Check if we have enough frames
        if end_frame > len(features):
            # Not enough frames
            available = len(features) - start_frame
            if available < self.min_audio_frames:  # Use min threshold
                return None, False
            # Pad with zeros to reach expected count
            window = features[start_frame:]
            pad_length = expected_frames - len(window)
            window = np.pad(window, ((0, pad_length), (0, 0)), mode='constant')
            return window, True
        
        # Extract exact frames
        window = features[start_frame:end_frame]
        
        return window, True
    
    def _build_session_windows(self, session_id, dataset_name, proc_root, transcript_path):
        """
        Build windows using FRAME-BASED sliding (not time-based masking).
        Guarantees exact frame counts per window.
        """
        # Get segments and labels
        t0 = self.t0_dict.get(session_id, 0.0)
        segments = self._load_participant_segments(transcript_path, t0)
        
        if not segments:
            return []
        
        phq_score, phq_binary, gender, fold = self._get_session_labels(session_id, dataset_name)
        if phq_binary is None:
            return []
        
        # Load speech-remapped features (continuous arrays)
        audio_features = self._load_audio_features(session_id, proc_root, segments)
        visual_features = self._load_visual_features(session_id, proc_root, segments)
        
        if audio_features is None and visual_features is None:
            return []
        
        # Determine max possible windows based on available frames
        max_audio_windows = 0
        max_visual_windows = 0
        
        if audio_features is not None:
            max_audio_windows = (len(audio_features) - self.expected_audio_frames) // self.audio_stride_frames + 1
        
        if visual_features is not None:
            max_visual_windows = (len(visual_features) - self.expected_visual_frames) // self.visual_stride_frames + 1
        
        # Use the minimum to ensure both modalities can be extracted
        max_windows = min(max_audio_windows, max_visual_windows) if (audio_features is not None and visual_features is not None) else \
                     max(max_audio_windows, max_visual_windows)
        
        if max_windows <= 0:
            return []
        
        # Generate windows
        windows = []
        
        for w_idx in range(max_windows):
            # Calculate frame indices
            audio_start = w_idx * self.audio_stride_frames
            visual_start = w_idx * self.visual_stride_frames
            
            # Extract audio window
            audio_window = None
            audio_valid = False
            if audio_features is not None:
                audio_window, audio_valid = self._extract_fixed_window(
                    audio_features, audio_start, self.expected_audio_frames
                )
            
            # Extract visual window
            visual_window = None
            visual_valid = False
            if visual_features is not None:
                visual_window, visual_valid = self._extract_fixed_window(
                    visual_features, visual_start, self.expected_visual_frames
                )
            
            # Skip if neither modality is valid
            if not audio_valid and not visual_valid:
                break  # No more valid windows
            
            # Verify frame counts (CRITICAL CHECK)
            if audio_window is not None and audio_window.shape[0] != self.expected_audio_frames:
                print(f"    WARNING: Audio frame mismatch at w{w_idx}: {audio_window.shape[0]} != {self.expected_audio_frames}")
                continue
            
            if visual_window is not None and visual_window.shape[0] != self.expected_visual_frames:
                print(f"    WARNING: Visual frame mismatch at w{w_idx}: {visual_window.shape[0]} != {self.expected_visual_frames}")
                continue
            
            # Create window metadata
            window_data = {
                'session': session_id,
                'dataset': dataset_name,
                'window_idx': w_idx,
                'audio_start_frame': audio_start if audio_valid else -1,
                'visual_start_frame': visual_start if visual_valid else -1,
                'y_reg': phq_score if phq_score is not None else 0.0,
                'y_bin': phq_binary,
                'gender': gender if gender is not None else 'Unknown',
                'fold': fold if fold is not None else 'Unknown',
                'audio_frames': audio_window.shape[0] if audio_window is not None else 0,
                'visual_frames': visual_window.shape[0] if visual_window is not None else 0,
                'audio_raw': audio_window.flatten() if audio_window is not None else None,
                'visual_raw': visual_window.flatten() if visual_window is not None else None
            }
            
            windows.append(window_data)
        
        return windows
    
    def build_dataset_cache(self, dataset_name='DAIC-WOZ'):
        """Build unified cache"""
        proc_root = self.config['outputs']['processed_root']
        transcript_dir = self.config['paths']['daic_woz']['transcript_dir']
        
        transcript_files = sorted(glob.glob(os.path.join(transcript_dir, "*_TRANSCRIPT.csv")))
        
        print(f"\n{'='*60}")
        print(f"Building {dataset_name} Cache")
        print(f"{'='*60}")
        print(f"Sessions: {len(transcript_files)}\n")
        
        all_windows = []
        session_stats = []
        
        for transcript_path in tqdm(transcript_files, desc="Processing"):
            session_id = os.path.basename(transcript_path).split('_')[0]
            
            try:
                windows = self._build_session_windows(
                    session_id, dataset_name, proc_root, transcript_path
                )
                
                if windows:
                    all_windows.extend(windows)
                    
                    # Verify frame counts
                    audio_counts = [w['audio_frames'] for w in windows if w['audio_frames'] > 0]
                    visual_counts = [w['visual_frames'] for w in windows if w['visual_frames'] > 0]
                    
                    session_stats.append({
                        'session': session_id,
                        'windows': len(windows),
                        'audio_min': min(audio_counts) if audio_counts else 0,
                        'audio_max': max(audio_counts) if audio_counts else 0,
                        'visual_min': min(visual_counts) if visual_counts else 0,
                        'visual_max': max(visual_counts) if visual_counts else 0
                    })
            
            except Exception as e:
                print(f"\n  ✗ {session_id}: {e}")
        
        if not all_windows:
            print("\n✗ No valid windows generated!")
            return None
        
        df = pd.DataFrame(all_windows)
        
        # Verification report
        print(f"\n{'='*60}")
        print(f"Dataset Statistics")
        print(f"{'='*60}")
        print(f"Total windows: {len(df)}")
        print(f"Total sessions: {df['session'].nunique()}")
        print(f"Depression ratio: {df['y_bin'].mean():.3f} ({df['y_bin'].sum():.0f}/{len(df)})")
        
        # Frame count verification
        if 'audio_frames' in df.columns:
            audio_df = df[df['audio_frames'] > 0]
            if len(audio_df) > 0:
                print(f"\nAudio frames:")
                print(f"  Expected: {self.expected_audio_frames}")
                print(f"  Actual range: [{audio_df['audio_frames'].min()}, {audio_df['audio_frames'].max()}]")
                print(f"  All exact: {(audio_df['audio_frames'] == self.expected_audio_frames).all()}")
        
        if 'visual_frames' in df.columns:
            visual_df = df[df['visual_frames'] > 0]
            if len(visual_df) > 0:
                print(f"\nVisual frames:")
                print(f"  Expected: {self.expected_visual_frames}")
                print(f"  Actual range: [{visual_df['visual_frames'].min()}, {visual_df['visual_frames'].max()}]")
                print(f"  All exact: {(visual_df['visual_frames'] == self.expected_visual_frames).all()}")
        
        # Fold distribution
        print(f"\nFold distribution:")
        for fold in ['train', 'dev', 'test']:
            fold_df = df[df['fold'] == fold]
            if len(fold_df) > 0:
                print(f"  {fold}: {len(fold_df)} windows from {fold_df['session'].nunique()} sessions")
        
        return df
    
    def apply_pca(self, df, fit_on_train_only=True):
        """
        Apply PCA with NO DATA LEAKAGE.
        Fits on train fold only, transforms all folds.
        """
        print(f"\n{'='*60}")
        print(f"Applying PCA (fit_on_train_only={fit_on_train_only})")
        print(f"{'='*60}")
        
        df_pca = df.copy()
        
        # Audio PCA
        if 'audio_raw' in df.columns and self.pca_components.get('audio', 0) > 0:
            audio_df = df[df['audio_raw'].notna()].copy()
            
            if len(audio_df) > 0:
                # Prepare data
                audio_matrix = np.vstack(audio_df['audio_raw'].values)
                
                # Fit scaler and PCA on train only
                if fit_on_train_only:
                    train_mask = audio_df['fold'] == 'train'
                    train_data = audio_matrix[train_mask]
                    
                    if len(train_data) == 0:
                        print("  Warning: No train data for audio PCA")
                        return df_pca
                    
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(train_data)
                    
                    n_comp = min(self.pca_components['audio'], train_scaled.shape[1])
                    pca = PCA(n_components=n_comp)
                    pca.fit(train_scaled)
                    
                    # Transform all data
                    all_scaled = scaler.transform(audio_matrix)
                    audio_pca = pca.transform(all_scaled)
                    
                    print(f"  Audio PCA: {audio_matrix.shape[1]} -> {n_comp} dims")
                    print(f"    Fitted on: {len(train_data)} train samples")
                    print(f"    Transformed: {len(audio_matrix)} total samples")
                    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
                else:
                    # Fit on all (for debugging/comparison only)
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(audio_matrix)
                    
                    n_comp = min(self.pca_components['audio'], scaled.shape[1])
                    pca = PCA(n_components=n_comp)
                    audio_pca = pca.fit_transform(scaled)
                    
                    print(f"  Audio PCA: {audio_matrix.shape[1]} -> {n_comp} dims (ALL DATA)")
                
                # Convert each row to list and assign
                audio_pca_list = [row.tolist() for row in audio_pca]
                df_pca.loc[df['audio_raw'].notna(), 'audio_pca'] = audio_pca_list
                
                self.pca_audio_model = {'scaler': scaler, 'pca': pca}
        
        # Visual PCA (same logic)
        if 'visual_raw' in df.columns and self.pca_components.get('visual', 0) > 0:
            visual_df = df[df['visual_raw'].notna()].copy()
            
            if len(visual_df) > 0:
                visual_matrix = np.vstack(visual_df['visual_raw'].values)
                
                if fit_on_train_only:
                    train_mask = visual_df['fold'] == 'train'
                    train_data = visual_matrix[train_mask]
                    
                    if len(train_data) == 0:
                        print("  Warning: No train data for visual PCA")
                        return df_pca
                    
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(train_data)
                    
                    n_comp = min(self.pca_components['visual'], train_scaled.shape[1])
                    pca = PCA(n_components=n_comp)
                    pca.fit(train_scaled)
                    
                    all_scaled = scaler.transform(visual_matrix)
                    visual_pca = pca.transform(all_scaled)
                    
                    print(f"  Visual PCA: {visual_matrix.shape[1]} -> {n_comp} dims")
                    print(f"    Fitted on: {len(train_data)} train samples")
                    print(f"    Transformed: {len(visual_matrix)} total samples")
                    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
                else:
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(visual_matrix)
                    
                    n_comp = min(self.pca_components['visual'], scaled.shape[1])
                    pca = PCA(n_components=n_comp)
                    visual_pca = pca.fit_transform(scaled)
                    
                    print(f"  Visual PCA: {visual_matrix.shape[1]} -> {n_comp} dims (ALL DATA)")
                
                # Convert each row to list and assign
                visual_pca_list = [row.tolist() for row in visual_pca]
                df_pca.loc[df['visual_raw'].notna(), 'visual_pca'] = visual_pca_list
                
                self.pca_visual_model = {'scaler': scaler, 'pca': pca}
        
        return df_pca
    
    def save_cache(self, df, output_dir):
        """Save cache efficiently (pickle + npz)"""
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_name = df['dataset'].iloc[0] if len(df) > 0 else 'unknown'
        base_name = f"{dataset_name}_win{self.window_duration}s"
        
        # Save full dataframe as pickle
        pickle_path = os.path.join(output_dir, f"{base_name}_cache.pkl")
        df.to_pickle(pickle_path)
        print(f"\nSaved: {pickle_path}")
        
        # Save metadata only as CSV (lightweight)
        metadata_cols = ['session', 'dataset', 'window_idx', 'y_reg', 'y_bin', 
                        'gender', 'fold', 'audio_frames', 'visual_frames']
        df_meta = df[[col for col in metadata_cols if col in df.columns]]
        csv_path = os.path.join(output_dir, f"{base_name}_metadata.csv")
        df_meta.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
        # Save PCA models
        if hasattr(self, 'pca_audio_model'):
            pca_path = os.path.join(output_dir, f"{base_name}_pca_audio.pkl")
            with open(pca_path, 'wb') as f:
                pickle.dump(self.pca_audio_model, f)
            print(f"Saved: {pca_path}")
        
        if hasattr(self, 'pca_visual_model'):
            pca_path = os.path.join(output_dir, f"{base_name}_pca_visual.pkl")
            with open(pca_path, 'wb') as f:
                pickle.dump(self.pca_visual_model, f)
            print(f"Saved: {pca_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--dataset', default='DAIC-WOZ')
    parser.add_argument('--window_durations', nargs='+', type=float, default=[2.0, 4.0, 10.0])
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--min_frames_ratio', type=float, default=0.9)
    parser.add_argument('--pca_audio', type=int, default=50)
    parser.add_argument('--pca_visual', type=int, default=50)
    parser.add_argument('--output_dir', default='cache/unified')
    parser.add_argument('--no_train_only_pca', action='store_true',
                       help='Fit PCA on all data (not recommended - causes data leakage)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Process each window duration
    all_results = []
    
    for window_duration in args.window_durations:
        print(f"\n{'='*80}")
        print(f"PROCESSING WINDOW DURATION: {window_duration}s")
        print(f"{'='*80}")
        
        # Create builder
        builder = UnifiedWindowCacheBuilder(
            config=config,
            window_duration=window_duration,
            overlap_ratio=args.overlap,
            min_frames_ratio=args.min_frames_ratio,
            pca_components={'audio': args.pca_audio, 'visual': args.pca_visual}
        )
        
        # Build cache
        df = builder.build_dataset_cache(dataset_name=args.dataset)
        
        if df is not None and len(df) > 0:
            # Apply PCA (train-only by default)
            df_pca = builder.apply_pca(df, fit_on_train_only=not args.no_train_only_pca)
            
            # Save cache
            output_subdir = os.path.join(args.output_dir, f"win{window_duration}s")
            builder.save_cache(df_pca, output_subdir)
            
            # Collect results
            all_results.append({
                'duration': window_duration,
                'total_windows': len(df_pca),
                'sessions': df_pca['session'].nunique(),
                'depression_ratio': df_pca['y_bin'].mean()
            })
        else:
            print(f"\n✗ No valid windows for {window_duration}s")
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        for res in all_results:
            print(f"Duration {res['duration']}s: {res['total_windows']} windows, "
                  f"{res['sessions']} sessions, depression={res['depression_ratio']:.3f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()