"""
Train Pitch Type Classifier

MLB Statcast 데이터를 다운로드하여 구종 분류 모델(XGBoost)을 학습합니다.
`pybaseball` 라이브러리를 사용하여 최근 1년간의 메이저리그 투구 데이터를 수집합니다.

Usage:
    python scripts/train_pitch_type.py --year 2024 --limit 100000
"""

import argparse
from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd

# ML Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# MLB Data
try:
    from pybaseball import statcast
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def fetch_mlb_data(start_date: str, end_date: str, limit: int = None) -> pd.DataFrame:
    if not PYBASEBALL_AVAILABLE:
        raise ImportError("pybaseball not installed. Run `pip install pybaseball`")
        
    log.info(f"Fetching MLB Statcast data from {start_date} to {end_date}...")
    
    # Fetch data
    df = statcast(start_dt=start_date, end_dt=end_date)
    
    if limit and len(df) > limit:
        df = df.sample(limit)
        
    log.info(f"Downloaded {len(df)} pitches.")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    필요한 Feature 선택 및 전처리
    Features: release_speed (mph), pfx_x (ft), pfx_z (ft)
    Target: pitch_type
    """
    # 1. 결측치 제거
    cols = ['pitch_type', 'release_speed', 'pfx_x', 'pfx_z']
    df = df.dropna(subset=cols)
    
    # 2. 주요 구종만 필터링 (너무 희귀한 구종 제외)
    major_types = ['FF', 'SL', 'CU', 'CH', 'SI', 'FC', 'FS', 'KC']
    df = df[df['pitch_type'].isin(major_types)]
    
    # 3. 단위 변환 (Imperial -> Metric) -> 우리 시스템은 Metric(km/h, cm) 사용
    # release_speed: mph -> km/h (1 mph = 1.60934 km/h)
    # pfx_x, pfx_z: ft -> cm (1 ft = 30.48 cm)
    
    df['velocity_kph'] = df['release_speed'] * 1.60934
    df['break_x_cm'] = df['pfx_x'] * 30.48
    df['break_z_cm'] = df['pfx_z'] * 30.48
    
    log.info(f"Preprocessed {len(df)} samples.")
    return df

def train_model(df: pd.DataFrame, output_path: str):
    # Features & Target
    X = df[['velocity_kph', 'break_x_cm', 'break_z_cm']].values
    y_raw = df['pitch_type'].values
    
    # Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost Model
    log.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='multi:softprob',
        num_class=len(le.classes_),
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    log.info(f"Accuracy: {acc:.4f}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save Artifacts
    save_data = {
        'model': model,
        'encoder': le,
        'features': ['velocity_kph', 'break_x_cm', 'break_z_cm']
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    log.info(f"Model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2024-04-01', help='Start Date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-10-01', help='End Date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='weights/pitch_classifier.pkl')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--smoke-test', action='store_true', help='Use dummy data for testing pipeline')
    
    args = parser.parse_args()
    
    if args.smoke_test:
        log.info("Running SMOKE TEST with dummy data...")
        # create dummy dataframe
        data = {
            'pitch_type': ['FF', 'SL', 'CU'] * 100,
            'release_speed': np.random.normal(90, 5, 300),
            'pfx_x': np.random.normal(0, 0.5, 300),
            'pfx_z': np.random.normal(1, 0.5, 300)
        }
        df = pd.DataFrame(data)
    else:
        df = fetch_mlb_data(args.start, args.end, args.limit)
        
    df = preprocess_data(df)
    train_model(df, args.output)

if __name__ == "__main__":
    main()
