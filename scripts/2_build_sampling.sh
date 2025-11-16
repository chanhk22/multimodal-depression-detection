python -m precache.window_sampling \
    --config configs/default.yaml \
    --dataset DAIC-WOZ \
    --window_durations 2 4 10 \
    --overlap 0.5 \
    --pca_audio 50 \
    --pca_visual 50 \
    --output_dir cache/unified

    # 기본 실행 (train-only PCA)
#python window_sampling.py --window_durations 2 4 10

# 특정 window만
#python window_sampling.py --window_durations 2

# 최소 프레임 비율 조정 (더 엄격하게)
#python window_sampling.py --min_frames_ratio 0.95

# PCA 비활성화
#python window_sampling.py --pca_audio 0 --pca_visual 0

# 모든 데이터로 PCA (비권장 - data leakage)
#python window_sampling.py --no_train_only_pca
