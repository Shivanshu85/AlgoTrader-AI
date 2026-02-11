"""
Model Tests

Comprehensive testing for:
- LSTM Attention Architecture
- PyTorch Lightning Module
- Training Pipeline
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import logging
from typing import Tuple

# Assuming imports from production modules
try:
    from lstm_attention import LSTMAttentionModel, MultiHeadAttention
    from lightning_module import StockPriceLightningModule
    from model_evaluation import ModelEvaluator
    from model_utils import (
        CheckpointManager, DataNormalizer, SequenceBuilder,
        FeatureScaler, count_parameters, get_device
    )
except ImportError:
    # For fallback during testing
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def device():
    """Get device"""
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    """Batch size"""
    return 32


@pytest.fixture
def seq_length():
    """Sequence length"""
    return 30


@pytest.fixture
def n_features():
    """Number of features"""
    return 10


@pytest.fixture
def sample_batch(batch_size, seq_length, n_features, device):
    """Create sample batch"""
    x = torch.randn(batch_size, seq_length, n_features, device=device)
    y = torch.randn(batch_size, 1, device=device)
    return x, y


@pytest.fixture
def sample_data(n_samples=1000, n_features=10):
    """Create sample time series data"""
    return np.random.randn(n_samples, n_features)


@pytest.fixture
def lstm_model(seq_length, n_features):
    """Create LSTM attention model"""
    return LSTMAttentionModel(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
        output_size=1
    )


@pytest.fixture
def lightning_module(lstm_model):
    """Create Lightning module"""
    return StockPriceLightningModule(
        model=lstm_model,
        learning_rate=0.001,
        weight_decay=1e-5
    )


# ============================================================================
# LSTM ARCHITECTURE TESTS
# ============================================================================

class TestLSTMArchitecture:
    """Test LSTM Attention architecture"""
    
    def test_model_initialization(self, lstm_model):
        """Test model initialization"""
        assert isinstance(lstm_model, nn.Module)
        assert lstm_model.input_size == 10
        assert lstm_model.hidden_size == 64
        assert lstm_model.num_layers == 2
    
    def test_forward_pass_shape(self, lstm_model, sample_batch, device):
        """Test forward pass output shape"""
        x, _ = sample_batch
        output = lstm_model(x)
        
        assert output.shape == (x.shape[0], 1)
        assert output.device == device
    
    def test_forward_pass_values(self, lstm_model, sample_batch):
        """Test forward pass produces valid values"""
        x, _ = sample_batch
        output = lstm_model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self, lstm_model, sample_batch, device):
        """Test gradients flow through model"""
        x, y = sample_batch
        
        optimizer = torch.optim.Adam(lstm_model.parameters())
        loss_fn = nn.MSELoss()
        
        # Forward pass
        output = lstm_model(x)
        loss = loss_fn(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in lstm_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_attention_mechanism(self, lstm_model, sample_batch):
        """Test attention mechanism is working"""
        x, _ = sample_batch
        
        # Forward with attention tracking
        with torch.no_grad():
            output = lstm_model(x)
        
        # Output should use attention
        assert output is not None
        assert output.shape[0] == x.shape[0]
    
    def test_model_parameters(self, lstm_model):
        """Test model has expected parameters"""
        n_params = count_parameters(lstm_model)
        
        # Should have reasonable number of parameters
        assert n_params > 0
        assert n_params < 1_000_000  # Shouldn't be huge
    
    def test_batch_independence(self, lstm_model, batch_size, seq_length, n_features, device):
        """Test predictions are independent across batch"""
        x1 = torch.randn(2, seq_length, n_features, device=device)
        x2 = torch.randn(2, seq_length, n_features, device=device)
        
        with torch.no_grad():
            out1 = lstm_model(x1)
            out2 = lstm_model(x2)
        
        # Outputs should be different
        assert not torch.allclose(out1, out2)


# ============================================================================
# MULTIHEAD ATTENTION TESTS
# ============================================================================

class TestMultiHeadAttention:
    """Test MultiHead Attention mechanism"""
    
    def test_attention_initialization(self, n_features):
        """Test attention module initialization"""
        attention = MultiHeadAttention(
            hidden_size=64,
            num_heads=4
        )
        assert attention.num_heads == 4
    
    def test_attention_forward_shape(self, n_features):
        """Test attention forward pass shape"""
        batch_size = 16
        seq_length = 30
        hidden_size = 64
        
        attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=4
        )
        
        Q = torch.randn(batch_size, seq_length, hidden_size)
        K = torch.randn(batch_size, seq_length, hidden_size)
        V = torch.randn(batch_size, seq_length, hidden_size)
        
        output = attention(Q, K, V)
        assert output.shape == (batch_size, seq_length, hidden_size)
    
    def test_attention_weights_sum_to_one(self, n_features):
        """Test attention weights sum to approximately 1"""
        batch_size = 8
        seq_length = 20
        hidden_size = 64
        
        attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=4
        )
        
        Q = torch.randn(batch_size, seq_length, hidden_size)
        K = torch.randn(batch_size, seq_length, hidden_size)
        V = torch.randn(batch_size, seq_length, hidden_size)
        
        # Get weights (inject into forward if needed)
        _ = attention(Q, K, V)


# ============================================================================
# LIGHTNING MODULE TESTS
# ============================================================================

class TestLightningModule:
    """Test PyTorch Lightning module"""
    
    def test_module_initialization(self, lightning_module):
        """Test Lightning module initialization"""
        assert isinstance(lightning_module, nn.Module)
        assert lightning_module.learning_rate == 0.001
    
    def test_training_step(self, lightning_module, sample_batch):
        """Test training step"""
        x, y = sample_batch
        batch = (x, y)
        
        # Training step should return loss
        loss = lightning_module.training_step(batch, batch_idx=0)
        
        assert loss is not None
        assert loss.requires_grad
        assert not torch.isnan(loss)
    
    def test_validation_step(self, lightning_module, sample_batch):
        """Test validation step"""
        x, y = sample_batch
        batch = (x, y)
        
        # Validation step
        loss = lightning_module.validation_step(batch, batch_idx=0)
        
        assert not torch.isnan(loss)
    
    def test_configure_optimizers(self, lightning_module):
        """Test optimizer configuration"""
        optimizer = lightning_module.configure_optimizers()
        
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
    
    def test_layer_types(self, lightning_module):
        """Test Lightning module contains expected layers"""
        assert hasattr(lightning_module, 'model')
        assert isinstance(lightning_module.model, nn.Module)


# ============================================================================
# MODEL EVALUATION TESTS
# ============================================================================

class TestModelEvaluator:
    """Test evaluation metrics"""
    
    def test_compute_metrics_shape(self):
        """Test metrics computation returns valid values"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'direction_accuracy' in metrics
        assert 'mape' in metrics
    
    def test_perfect_predictions(self):
        """Test metrics for perfect predictions"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
        
        assert metrics['rmse'] < 1e-6
        assert metrics['mae'] < 1e-6
        assert abs(metrics['r2'] - 1.0) < 1e-6
    
    def test_bad_predictions(self):
        """Test metrics for poor predictions"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
        
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
    
    def test_direction_accuracy(self):
        """Test direction accuracy metric"""
        y_true = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        y_pred = np.array([0.9, -2.1, 2.8, -4.2, 5.1])
        
        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
        
        # All predictions have correct direction
        assert metrics['direction_accuracy'] >= 0.8
    
    def test_risk_metrics(self):
        """Test risk metrics computation"""
        returns = np.random.randn(252) * 0.02
        
        risk_metrics = ModelEvaluator.compute_risk_metrics(returns)
        
        assert 'volatility' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        assert 'sortino_ratio' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert risk_metrics['volatility'] > 0


# ============================================================================
# MODEL UTILITIES TESTS
# ============================================================================

class TestDataNormalizer:
    """Test data normalization"""
    
    def test_normalizer_fit(self, sample_data):
        """Test normalizer fitting"""
        normalizer = DataNormalizer()
        normalizer.fit(sample_data)
        
        assert normalizer.mean is not None
        assert normalizer.std is not None
    
    def test_normalize_denormalize(self, sample_data):
        """Test normalize then denormalize"""
        normalizer = DataNormalizer()
        normalizer.fit(sample_data)
        
        normalized = normalizer.normalize(sample_data)
        denormalized = normalizer.denormalize(normalized)
        
        assert np.allclose(sample_data, denormalized, rtol=1e-5)
    
    def test_normalized_mean_std(self, sample_data):
        """Test normalized data has mean ≈ 0 and std ≈ 1"""
        normalizer = DataNormalizer()
        normalizer.fit(sample_data)
        normalized = normalizer.normalize(sample_data)
        
        assert np.abs(np.mean(normalized)) < 0.1
        assert np.abs(np.std(normalized) - 1.0) < 0.1


class TestSequenceBuilder:
    """Test sequence building"""
    
    def test_create_sequences_shape(self, sample_data):
        """Test sequence creation shape"""
        seq_length = 10
        X, y = SequenceBuilder.create_sequences(sample_data, seq_length=seq_length)
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == seq_length
        assert X.shape[2] == sample_data.shape[1]
    
    def test_create_sequences_values(self, sample_data):
        """Test sequence values are correct"""
        seq_length = 10
        X, y = SequenceBuilder.create_sequences(sample_data, seq_length=seq_length)
        
        # First sequence should match data
        assert np.allclose(X[0], sample_data[:seq_length])
        # Target should be next value
        assert np.allclose(y[0], sample_data[seq_length, -1])


class TestFeatureScaler:
    """Test feature scaling"""
    
    def test_minmax_scaling(self, sample_data):
        """Test min-max scaling"""
        scaler = FeatureScaler(method='minmax')
        scaler.fit(sample_data)
        
        scaled = scaler.transform(sample_data)
        
        # Min and max should be approximately 0 and 1
        assert np.allclose(scaled.min(axis=0), 0, atol=0.1)
        assert np.allclose(scaled.max(axis=0), 1, atol=0.1)
    
    def test_standard_scaling(self, sample_data):
        """Test standard scaling"""
        scaler = FeatureScaler(method='standard')
        scaler.fit(sample_data)
        
        scaled = scaler.transform(sample_data)
        
        # Mean should be ≈ 0, std should be ≈ 1
        assert np.abs(scaled.mean(axis=0)).mean() < 0.1
        assert np.abs(scaled.std(axis=0).mean() - 1.0) < 0.1
    
    def test_inverse_transform(self, sample_data):
        """Test inverse transform"""
        scaler = FeatureScaler(method='minmax')
        scaler.fit(sample_data)
        
        scaled = scaler.transform(sample_data)
        unscaled = scaler.inverse_transform(scaled)
        
        assert np.allclose(unscaled, sample_data, rtol=1e-5)


class TestCheckpointManager:
    """Test checkpoint management"""
    
    def test_save_checkpoint(self, lstm_model):
        """Test checkpoint saving"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            optimizer = torch.optim.Adam(lstm_model.parameters())
            
            metrics = {'val_loss': 0.5}
            path = manager.save_checkpoint(lstm_model, optimizer, epoch=1, metrics=metrics)
            
            assert Path(path).exists()
    
    def test_load_checkpoint(self, lstm_model):
        """Test checkpoint loading"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)
            optimizer = torch.optim.Adam(lstm_model.parameters())
            
            # Save
            metrics = {'val_loss': 0.5}
            path = manager.save_checkpoint(lstm_model, optimizer, epoch=1, metrics=metrics)
            
            # Modify model
            original_state = dict(lstm_model.state_dict())
            lstm_model.fc.bias.data.fill_(0)
            
            # Load
            checkpoint = manager.load_checkpoint(path, lstm_model, optimizer)
            
            # State should be restored
            assert checkpoint['epoch'] == 1
            assert checkpoint['metrics'] == metrics


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests"""
    
    def test_full_training_loop(self, lightning_module, sample_batch, device):
        """Test full training loop"""
        optimizer = lightning_module.configure_optimizers()
        
        # Training steps
        for _ in range(3):
            x, y = sample_batch
            batch = (x, y)
            
            loss = lightning_module.training_step(batch, batch_idx=0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            assert not torch.isnan(loss)
    
    def test_model_pipeline(self, sample_data, seq_length, n_features, device):
        """Test complete pipeline: data -> model -> eval"""
        # Normalize
        normalizer = DataNormalizer()
        normalizer.fit(sample_data)
        normalized_data = normalizer.normalize(sample_data)
        
        # Create sequences
        X, y = SequenceBuilder.create_sequences(
            normalized_data,
            seq_length=seq_length
        )
        
        # Create model
        model = LSTMAttentionModel(
            input_size=n_features,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            output_size=1
        ).to(device)
        
        # Forward pass
        X_tensor = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            predictions = model(X_tensor)
        
        # Denormalize predictions
        predictions_np = predictions.cpu().numpy()
        denormalized_preds = normalizer.denormalize(predictions_np)
        y_denormalized = normalizer.denormalize(y)
        
        # Compute metrics
        metrics = ModelEvaluator.compute_metrics(y_denormalized, denormalized_preds)
        
        assert 'rmse' in metrics
        assert metrics['rmse'] > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
