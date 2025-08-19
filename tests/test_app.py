import pytest
import pandas as pd
import numpy as np
import pickle
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import preprocess_data, get_strategy_explanation, load_model

class TestDataProcessing:
    """Test data preprocessing functions"""
    
    def test_preprocess_data_removes_unwanted_columns(self):
        """Test that unwanted columns are properly removed"""
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02'],
            'LLL1 Index': [1.0, 2.0],
            'VIX Index': [20.0, 25.0],
            'MXWO Index': [100.0, 105.0]
        })
        
        result = preprocess_data(df.copy())
        assert 'LLL1 Index' not in result.columns
        
    def test_preprocess_data_creates_lagged_features(self):
        """Test that lagged features are created correctly"""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'VIX Index': range(10),
            'MXWO Index': range(10, 20)
        })
        
        result = preprocess_data(df)
        
        # Check that lagged columns exist
        assert 'VIX Index_lag_1' in result.columns
        assert 'VIX Index_lag_2' in result.columns
        assert 'VIX Index_lag_3' in result.columns
        assert 'MXWO Index_lag_1' in result.columns
        
    def test_preprocess_data_drops_nan_rows(self):
        """Test that rows with NaN values are dropped"""
        df = pd.DataFrame({
            'VIX Index': range(5),
            'MXWO Index': range(5, 10)
        })
        
        result = preprocess_data(df)
        
        # After creating 3 lags, first 3 rows should be dropped
        assert len(result) == 2
        assert not result.isnull().any().any()

class TestStrategyExplanation:
    """Test strategy explanation function"""
    
    def test_high_risk_strategy(self):
        """Test high risk strategy generation"""
        result = get_strategy_explanation("high", 35.0, 1.5, 110.0, 85.0)
        
        assert "Defensive Strategy Recommended" in result['summary']
        assert result['rationale'] is not None
        assert len(result['actions']) == 4
        assert "12.0%" in result['actions'][0]  # Check potential savings
        
    def test_medium_risk_strategy(self):
        """Test medium risk strategy generation"""
        result = get_strategy_explanation("medium", 25.0, 0.5, 105.0, 50.0)
        
        assert "Balanced Approach Needed" in result['summary']
        assert "7.5%" in result['actions'][0]  # Check potential savings
        
    def test_low_risk_strategy(self):
        """Test low risk strategy generation"""
        result = get_strategy_explanation("low", 15.0, 0.2, 100.0, 20.0)
        
        assert "Growth Opportunities Present" in result['summary']
        assert result['timeframe'] is not None
        assert result['historical_accuracy'] is not None

class TestModelLoading:
    """Test model loading functionality"""
    
    @patch('pickle.load')
    @patch('builtins.open')
    def test_successful_model_loading(self, mock_open, mock_pickle_load):
        """Test successful model loading"""
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model
        
        result = load_model('test_model.pkl')
        
        assert result == mock_model
        mock_open.assert_called_once_with('test_model.pkl', 'rb')
        
    @patch('pickle.load')
    @patch('builtins.open')
    @patch('streamlit.error')
    def test_model_loading_with_backup(self, mock_st_error, mock_open, mock_pickle_load):
        """Test model loading with backup fallback"""
        # First call fails, second succeeds with backup
        mock_backup_model = Mock()
        mock_pickle_load.side_effect = [Exception("File not found"), mock_backup_model]
        
        # Mock open to handle both file attempts
        mock_open.side_effect = [Mock(), Mock()]
        
        result = load_model('test_model.pkl')
        
        assert result == mock_backup_model
        assert mock_open.call_count == 2

class TestDataValidation:
    """Test data validation and edge cases"""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes"""
        df = pd.DataFrame()
        
        with pytest.raises((IndexError, KeyError)):
            preprocess_data(df)
            
    def test_missing_required_columns(self):
        """Test handling of missing required columns"""
        df = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        
        with pytest.raises(KeyError):
            preprocess_data(df)

class TestPerformanceMetrics:
    """Test performance tracking functionality"""
    
    def test_strategy_explanation_performance(self):
        """Test that strategy explanation runs efficiently"""
        import time
        
        start_time = time.time()
        get_strategy_explanation("medium", 25.0, 0.5, 105.0, 50.0)
        execution_time = time.time() - start_time
        
        # Should complete in under 100ms
        assert execution_time < 0.1

if __name__ == "__main__":
    pytest.main([__file__])
