from typing import Dict, Union, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class MotorcyclePricePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, model_path: str = 'optimized_price_model_with_tf.joblib'):
        """Initialize preprocessor with model artifacts."""
        self.current_year = 2024
        try:
            model_artifacts = joblib.load(model_path)
            self.models = model_artifacts['models']
            self.tf_model = model_artifacts['tf_model']
            self.scaler = model_artifacts['scaler']
            self.weights = model_artifacts['weights']
            self.feature_columns = model_artifacts['feature_columns']
            self.categorical_columns = model_artifacts['categorical_columns']
            self.numerical_columns = model_artifacts['numerical_columns']
        except Exception as e:
            raise Exception(f"Error loading model artifacts: {str(e)}")

        # Required input features
        self.required_features = ['model', 'year', 'mileage', 'location', 'tax', 'seller_type']

        # Province mapping
        self.province_mapping = {
            'Jakarta': ['Jakarta', 'Jakarta Timur', 'Jakarta Barat', 'Jakarta Selatan', 
                       'Jakarta Utara', 'Jakarta Pusat'],
            'Jawa Barat': ['Bandung', 'Depok', 'Bekasi', 'Bogor', 'Cimahi', 'Cianjur', 
                          'Ciamis', 'Garut', 'Sukabumi', 'Karawang'],
            'Banten': ['Tangerang', 'Tangerang Selatan', 'Serang', 'Cilegon'],
            'Jawa Tengah': ['Semarang', 'Magelang', 'Klaten', 'Pemalang'],
            'Yogyakarta': ['Yogyakarta', 'Sleman', 'Bantul'],
            'Jawa Timur': ['Surabaya', 'Malang', 'Sidoarjo', 'Gresik', 'Kediri'],
            'Bali': ['Denpasar', 'Badung', 'Buleleng']
        }

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input data."""
        # Check required features
        missing_features = [f for f in self.required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Check year
        if df['year'].max() > self.current_year:
            raise ValueError(f"Year cannot be greater than {self.current_year}")

        # Check mileage
        if df['mileage'].min() < 0:
            raise ValueError("Mileage cannot be negative")

        # Check seller_type
        valid_seller_types = ['dealer', 'private']
        invalid_seller = df[~df['seller_type'].str.lower().isin(valid_seller_types)]
        if not invalid_seller.empty:
            raise ValueError(f"Seller type must be either 'dealer' or 'private'. Found invalid values: {invalid_seller['seller_type'].tolist()}")

    def _clean_mileage(self, mileage: Union[str, float, int]) -> float:
        """Clean and standardize mileage format."""
        if isinstance(mileage, (int, float)):
            return float(mileage)

        try:
            mileage = str(mileage).lower()
            if '-' in mileage:
                start, end = mileage.split('-')
                start = float(start.replace('km', '').replace(',', '').replace('.', '').strip())
                end = float(end.replace('km', '').replace(',', '').replace('.', '').strip())
                return (start + end) / 2
            return float(mileage.replace('km', '').replace(',', '').replace('.', '').strip())
        except:
            raise ValueError(f"Invalid mileage format: {mileage}")

    def _extract_engine_size(self, model: str) -> int:
        """Extract engine size from model name."""
        model = str(model).upper()
        if '160' in model:
            return 160
        elif '150' in model:
            return 150
        elif '125' in model:
            return 125
        return 110

    def _map_location_to_province(self, location: str) -> str:
        """Map location to province."""
        location = str(location).lower()
        for province, cities in self.province_mapping.items():
            if any(city.lower() in location for city in cities):
                return province
        return 'Others'

    def transform(self, data: Dict[str, Union[str, float, int]]) -> pd.DataFrame:
        """Transform input data for prediction."""
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        # Validate input
        self._validate_input(df)

        # Basic cleaning
        df['mileage'] = df['mileage'].apply(self._clean_mileage)

        # Create basic features
        df['age'] = self.current_year - df['year']
        df['engine_size'] = df['model'].apply(self._extract_engine_size)
        df['province'] = df['location'].apply(self._map_location_to_province)

        # Create age categories
        try:
            df['age_category'] = pd.qcut(df['age'], 4,
                                       labels=['new', 'medium_new', 'medium_old', 'old'])
        except ValueError:
            df['age_category'] = 'new' if df['age'].iloc[0] <= 2 else 'old'

        # Create numerical features
        df['age_squared'] = df['age'] ** 2
        df['mileage_squared'] = df['mileage'] ** 2
        df['price_per_cc'] = df['engine_size']
        df['mileage_per_age'] = df['mileage'] / (df['age'] + 1)
        df['engine_age_interaction'] = df['engine_size'] * np.exp(-df['age']/3)
        df['normalized_mileage'] = df['mileage'] / (df['age'] + 1)
        df['depreciation_factor'] = np.exp(-df['age']/5)

        # Market features
        df['is_abs'] = df['model'].str.contains('ABS', case=False, na=False).astype(int)
        df['is_cbs'] = df['model'].str.contains('CBS|ISS', case=False, na=False).astype(int)
        df['is_premium'] = ((df['engine_size'] >= 150) |
                          (df['model'].str.contains('ABS|CBS', case=False, na=False))).astype(int)

        # Create price and age segments
        df['price_segment'] = 'medium'  # Default for new predictions
        
        # Create dummies
        categorical_columns = ['province', 'age_category', 'price_segment', 'seller_type']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)

        # Ensure all training features exist
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Select and order columns
        X = df_encoded[self.feature_columns]

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)

        return X_scaled

    def predict(self, data: Dict[str, Union[str, float, int]]) -> Dict[str, Union[float, str]]:
        """Make price prediction."""
        try:
            # Transform features
            X = self.transform(data)

            # Make predictions with each model
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(X)[0]
                predictions[name] = pred

            # TensorFlow prediction
            tf_pred = self.tf_model.predict(X)[0][0]
            predictions['tf'] = tf_pred

            # Calculate ensemble prediction
            final_prediction = sum(self.weights[name] * predictions[name]
                                 for name in self.weights.keys())

            # Calculate prediction range
            confidence_interval = 0.1  # 10% margin
            price_range = {
                'lower': final_prediction * (1 - confidence_interval),
                'upper': final_prediction * (1 + confidence_interval)
            }

            return {
                'status': 'success',
                'predictions': {
                    'rf': predictions['rf'],
                    'xgb': predictions['xgb'],
                    'gbm': predictions['gbm'],
                    'tf': predictions['tf'],
                    'final': final_prediction,
                    'price_range': price_range
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for multiple entries."""
        results = []
        for _, row in df.iterrows():
            prediction = self.predict(row.to_dict())
            if prediction['status'] == 'success':
                results.append({
                    'input': row.to_dict(),
                    'predicted_price': prediction['predictions']['final'],
                    'price_range_low': prediction['predictions']['price_range']['lower'],
                    'price_range_high': prediction['predictions']['price_range']['upper']
                })
            else:
                results.append({
                    'input': row.to_dict(),
                    'error': prediction['message']
                })

        return pd.DataFrame(results)