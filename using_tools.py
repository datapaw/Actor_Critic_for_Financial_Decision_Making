"""
Model-Driven Tool Selection and Feature Engineering

This script:
1. Loads the trained ResNet model from agent_first.py
2. Makes predictions on S&P 500 data for different time horizons
3. Selects appropriate technical analysis tools based on model predictions
4. Generates engineered features for each time horizon
5. Saves comprehensive results to sp500_tools.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from tools import (
    create_features_10day,
    create_features_15day,
    create_features_30day,
    create_features_60day,
    create_default_features,
    scale_features,
    analyze_feature_importance
)

# Configuration
DATA_PATH = Path("data/dataset/sp500_prep.csv")
MODEL_PATH = Path("models/resnet_sp500.keras")
OUTPUT_PATH = Path("data/dataset/sp500_tools.csv")
TIME_FRAME_DAYS = 20
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class ModelDrivenToolSelector:
    """Load trained model and select tools based on predictions"""
    
    def __init__(self, model_path: Path, data_path: Path):
        """
        Initialize with model and data
        
        INPUT:
            model_path: Path to trained ResNet model
            data_path: Path to sp500_prep.csv
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.df = None
        self.X = None
        self.y = None
        self.predictions = None
        self.confidence = None
        
    def load_model(self):
        """Load the trained ResNet model"""
        print(f"Loading model from {self.model_path}...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        print(f"Model loaded. Input shape: {self.model.input_shape}")
        return self.model
    
    def load_data(self):
        """Load and prepare data"""
        print(f"\nLoading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, index_col=0)
        self.df.index = pd.to_datetime(self.df.index)
        print(f"Data loaded: {self.df.shape}")
        return self.df
    
    def prepare_sequences(self):
        """Prepare rolling windows for model input"""
        print(f"\nPreparing sequences with window size {TIME_FRAME_DAYS}...")
        
        # Clean data
        df_clean = self.df.dropna(subset=FEATURE_COLUMNS)
        
        sequences = []
        indices = []
        
        for i in range(TIME_FRAME_DAYS, len(df_clean)):
            window = df_clean.iloc[i - TIME_FRAME_DAYS : i][FEATURE_COLUMNS].to_numpy()
            sequences.append(window)
            indices.append(df_clean.index[i])
        
        self.X = np.stack(sequences)
        self.sequence_indices = pd.DatetimeIndex(indices)
        print(f"Prepared {len(self.X)} sequences")
        return self.X
    
    def make_predictions(self):
        """Make predictions using the trained model"""
        print(f"\nMaking predictions...")
        
        # Get probability predictions
        y_pred_proba = self.model.predict(self.X, verbose=0)
        self.predictions = np.argmax(y_pred_proba, axis=1)
        self.confidence = np.max(y_pred_proba, axis=1)
        
        print(f"Predictions made for {len(self.predictions)} samples")
        print(f"Average confidence: {self.confidence.mean():.4f}")
        print(f"Prediction distribution:")
        unique, counts = np.unique(self.predictions, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} ({c/len(self.predictions)*100:.1f}%)")
        
        return self.predictions, self.confidence
    
    def select_tools_by_prediction(self, prediction: int, confidence: float) -> dict:
        """
        Select appropriate tools based on model prediction and confidence
        
        INPUT:
            prediction: int, predicted target_period (0, 1, 2, 3)
                - 0: 60+ days
                - 1: 30 days
                - 2: 15 days
                - 3: 10 days
            confidence: float, prediction confidence (0-1)
        
        OUTPUT:
            dict with tool recommendations and feature set
        """
        tools_recommended = {
            'primary_tools': [],
            'secondary_tools': [],
            'feature_set': None,
            'time_horizon': None,
            'strategy': None,
            'confidence': confidence
        }
        
        # HIGH CONFIDENCE: Use specialized tools
        if confidence > 0.75:
            if prediction == 3:  # 10 days
                tools_recommended['time_horizon'] = '10_days'
                tools_recommended['primary_tools'] = ['RSI', 'MACD', 'Stochastic', 'ROC']
                tools_recommended['secondary_tools'] = ['EMA_5_12', 'Volume_SMA', 'Price_Patterns']
                tools_recommended['feature_set'] = 'features_10day'
                tools_recommended['strategy'] = 'Momentum-based - Quick entries/exits'
            
            elif prediction == 2:  # 15 days
                tools_recommended['time_horizon'] = '15_days'
                tools_recommended['primary_tools'] = ['EMA_Crossover', 'MACD_Histogram', 'Support_Resistance']
                tools_recommended['secondary_tools'] = ['Volume_Confirmation', 'ATR', 'Price_Patterns']
                tools_recommended['feature_set'] = 'features_15day'
                tools_recommended['strategy'] = 'Trend-following - Medium-term swings'
            
            elif prediction == 1:  # 30 days
                tools_recommended['time_horizon'] = '30_days'
                tools_recommended['primary_tools'] = ['SMA_Trend', 'Support_Resistance_30', 'Volatility_Regime']
                tools_recommended['secondary_tools'] = ['Volume_Trend', 'Price_Patterns', 'BB_Bands']
                tools_recommended['feature_set'] = 'features_30day'
                tools_recommended['strategy'] = 'Trend identification - Position holding'
            
            elif prediction == 0:  # 60 days
                tools_recommended['time_horizon'] = '60_days'
                tools_recommended['primary_tools'] = ['Long_Term_Trend', 'Hurst_Exponent', 'Mean_Reversion']
                tools_recommended['secondary_tools'] = ['Support_Resistance_60', 'Autocorrelation', 'Return_Momentum']
                tools_recommended['feature_set'] = 'features_60day'
                tools_recommended['strategy'] = 'Structural analysis - Long-term positioning'
        
        # MEDIUM CONFIDENCE: Use balanced combination
        elif confidence > 0.6:
            tools_recommended['primary_tools'] = ['MACD', 'EMA_Crossover', 'Support_Resistance']
            tools_recommended['secondary_tools'] = ['Volume_Confirmation', 'ATR', 'Price_Patterns']
            tools_recommended['feature_set'] = 'features_default'
            tools_recommended['strategy'] = 'Balanced - Use default feature set'
        
        # LOW CONFIDENCE: Use default/safest approach
        else:
            tools_recommended['primary_tools'] = ['RSI', 'Moving_Averages', 'Volatility']
            tools_recommended['secondary_tools'] = ['Volume', 'Support_Resistance', 'Price_Patterns']
            tools_recommended['feature_set'] = 'features_default'
            tools_recommended['strategy'] = 'Conservative - Low confidence, use broad analysis'
        
        return tools_recommended
    
    def generate_features_for_prediction(self, df: pd.DataFrame, prediction: int, confidence: float) -> pd.DataFrame:
        """Generate appropriate features based on model prediction"""
        
        tool_selection = self.select_tools_by_prediction(prediction, confidence)
        feature_set_name = tool_selection['feature_set']
        
        # Call appropriate feature generation function
        if feature_set_name == 'features_10day':
            features = create_features_10day(df)
        elif feature_set_name == 'features_15day':
            features = create_features_15day(df)
        elif feature_set_name == 'features_30day':
            features = create_features_30day(df)
        elif feature_set_name == 'features_60day':
            features = create_features_60day(df)
        else:  # features_default
            features = create_default_features(df)
        
        return features, tool_selection
    
    def run_full_pipeline(self):
        """Run complete analysis pipeline"""
        
        print("="*70)
        print("MODEL-DRIVEN TOOL SELECTION & FEATURE ENGINEERING")
        print("="*70)
        
        # Load model and data
        self.load_model()
        self.load_data()
        
        # Prepare sequences and make predictions
        self.prepare_sequences()
        self.make_predictions()
        
        # Generate results dataframe
        print("\n" + "="*70)
        print("GENERATING FEATURES AND TOOLS FOR EACH PREDICTION")
        print("="*70)
        
        results_list = []
        
        # For each prediction, generate appropriate features
        for idx, (date, pred, conf) in enumerate(zip(
            self.sequence_indices,
            self.predictions,
            self.confidence
        )):
            if idx % 50 == 0:
                print(f"Processing {idx}/{len(self.predictions)}...")
            
            # Get tool selection
            tool_selection = self.select_tools_by_prediction(pred, conf)
            
            # Generate features
            features, _ = self.generate_features_for_prediction(self.df, pred, conf)
            
            # Get the feature values at this date
            if date in features.index:
                feature_row = features.loc[date]
                
                # Create result row
                result_row = {
                    'date': date,
                    'model_prediction_period': pred,
                    'model_confidence': conf,
                    'time_horizon': tool_selection['time_horizon'],
                    'primary_tools': '|'.join(tool_selection['primary_tools']),
                    'secondary_tools': '|'.join(tool_selection['secondary_tools']),
                    'strategy': tool_selection['strategy'],
                }
                
                # Add feature values
                for col in feature_row.index:
                    if pd.notna(feature_row[col]):
                        result_row[f'feature_{col}'] = feature_row[col]
                
                # Add price data if available
                if date in self.df.index:
                    price_row = self.df.loc[date]
                    result_row['close'] = price_row.get('Close', np.nan)
                    result_row['volume'] = price_row.get('Volume', np.nan)
                    result_row['close_diff_next_day'] = price_row.get('close_diff_next_day', np.nan)
                
                results_list.append(result_row)
        
        # Create results dataframe
        results_df = pd.DataFrame(results_list)
        
        print(f"\nGenerated {len(results_df)} complete feature sets")
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_path: Path = None):
        """Save results to CSV"""
        if output_path is None:
            output_path = OUTPUT_PATH
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"Total records: {len(results_df)}")
        print(f"Columns: {len(results_df.columns)}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"{'='*70}")
        
        # Print summary statistics
        print("\nPREDICTION DISTRIBUTION:")
        pred_dist = results_df['model_prediction_period'].value_counts().sort_index()
        period_map = {0: '60_days', 1: '30_days', 2: '15_days', 3: '10_days'}
        for pred, count in pred_dist.items():
            print(f"  {period_map.get(pred, 'Unknown')}: {count} ({count/len(results_df)*100:.1f}%)")
        
        print("\nCONFIDENCE STATISTICS:")
        print(f"  Mean confidence: {results_df['model_confidence'].mean():.4f}")
        print(f"  Min confidence: {results_df['model_confidence'].min():.4f}")
        print(f"  Max confidence: {results_df['model_confidence'].max():.4f}")
        print(f"  Std confidence: {results_df['model_confidence'].std():.4f}")
        
        print("\nTOOL DISTRIBUTION:")
        tool_counts = {}
        for tools in results_df['primary_tools']:
            for tool in tools.split('|'):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count} times ({count/len(results_df)*100:.1f}%)")


def main():
    """Main entry point"""
    
    # Create analyzer
    analyzer = ModelDrivenToolSelector(MODEL_PATH, DATA_PATH)
    
    # Run pipeline
    results_df = analyzer.run_full_pipeline()
    
    # Save results
    analyzer.save_results(results_df)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Output file: {OUTPUT_PATH}")
    print("Each row contains:")
    print("  - Model prediction and confidence")
    print("  - Recommended tools for that prediction")
    print("  - Generated features from those tools")
    print("  - Original price data")


if __name__ == "__main__":
    main()
