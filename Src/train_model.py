import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEL TRAINING PIPELINE - PRODUCTION VERSION")
print("="*80)

# ====================== 1. Load Data ======================
print("📊 Loading cleaned data...")
df = pd.read_pickle('data/cleaned_data.pkl')
print(f"Dataset shape: {df.shape}")

# Remove extreme outliers (optional but recommended)
df = df[df['price'] < df['price'].quantile(0.99)]
print(f"After outlier removal: {df.shape}")

# ====================== 2. Feature Engineering ======================
print("🔨 Engineering features...")

def simplify_location(loc):
    loc = str(loc).lower()
    if any(x in loc for x in ['new cairo', '5th', 'fifth']):
        return 'New Cairo'
    elif any(x in loc for x in ['sheikh zayed', 'zayed']):
        return 'Sheikh Zayed'
    elif any(x in loc for x in ['north coast', 'ras al hekma', 'alamein']):
        return 'North Coast'
    elif any(x in loc for x in ['hurghada', 'gouna', 'red sea']):
        return 'Hurghada / Red Sea'
    elif 'october' in loc:
        return '6 October'
    else:
        return 'Other'

df['location_group'] = df['location'].apply(simplify_location)

# Safe features only (no leakage)
X = df[['location_group', 'type', 'payment_method', 'size_sqm', 'bedrooms_num', 'bathrooms']].copy()
y = df['price']

print(f"Features: {X.columns.tolist()}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ====================== 3. Train-Test Split ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ====================== 4. Preprocessor ======================
numeric_features = ['size_sqm', 'bedrooms_num', 'bathrooms']
categorical_features = ['location_group', 'type', 'payment_method']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# ====================== 5. Models ======================
models = {
    "Ridge Regression": Ridge(alpha=100),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.08, max_depth=6, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=7, subsample=0.85, random_state=42, n_jobs=-1)
}

results = {}

print("\n" + "="*80)
print("TRAINING MODELS WITH CROSS-VALIDATION")
print("="*80)

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Cross Validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"   CV R²: {cv_scores.mean():.4f}")
    
    # Fit on full train set
    pipeline.fit(X_train, y_train)
    
    # Predict on test
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'CV R²': round(cv_scores.mean(), 4),
        'Train R²': round(pipeline.score(X_train, y_train), 4),
        'Test R²': round(r2, 4),
        'Test RMSE': round(rmse, 0),
        'Test MAE': round(mae, 0)
    }
    
    print(f"   Test R²: {r2:.4f} | RMSE: {rmse:,.0f} EGP | MAE: {mae:,.0f} EGP")

# ====================== Save Best Model ======================
best_model_name = max(results, key=lambda x: results[x]['Test R²'])
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', models[best_model_name])
])
best_pipeline.fit(X_train, y_train)   # retrain on full train

os.makedirs('models', exist_ok=True)
with open('models/best_pipeline.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R²: {results[best_model_name]['Test R²']}")
print(f"   Test RMSE: {results[best_model_name]['Test RMSE']:,} EGP")
print(f"   Test MAE: {results[best_model_name]['Test MAE']:,} EGP")

print(f"\n✅ Best pipeline saved to 'models/best_pipeline.pkl'")

# Save comparison
results_df = pd.DataFrame(results).T
results_df.to_csv('reports/model_comparison.csv')
print("✅ Results saved to 'reports/model_comparison.csv'")

print("\n✅ Training pipeline completed successfully!")