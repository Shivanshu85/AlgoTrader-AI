# ADR 004: ML Framework - PyTorch vs TensorFlow vs JAX

**Status:** ACCEPTED  
**Date:** February 2026  
**ADR Number:** 004  

---

## Context

We need an ML framework for training LSTM models with:
- Support for custom layers (attention)
- Production deployment capabilities
- GPU optimization
- Reproducibility and experiment tracking
- Community support

---

## Options Evaluated

### Option 1: PyTorch (SELECTED)

**Pros:**
- ✅ Dynamic computation graphs (easier debugging)
- ✅ Excellent for research and production
- ✅ PyTorch Lightning reduces boilerplate
- ✅ Easier to implement attention mechanisms
- ✅ Superior ONNX support for deployment
- ✅ Larger ML community (especially Kaggle)
- ✅ Better distributed training stories

**Use:** Primary framework

### Option 2: TensorFlow

**Cons:**
- ❌ Static graphs (steep learning curve)
- ❌ Keras API is more restrictive
- ❌ More verbose for custom layers
- ❌ Heavier deployment footprint

### Option 3: JAX

**Cons:**
- ❌ Immature ecosystem
- ❌ Very steep learning curve
- ❌ Limited library support
- ❌ Tiny community

---

## Decision

**SELECTED: PyTorch + PyTorch Lightning**

### Rationale

```
Flexibility:      PyTorch > TensorFlow > JAX
Ease of Use:      PyTorch > TensorFlow > JAX
Production:       PyTorch ≈ TensorFlow > JAX
Community:        PyTorch > TensorFlow > JAX
Learning Curve:   PyTorch > TensorFlow > JAX (lower is better)
```

### Implementation

```python
# production/models/lstm.py

import torch
import torch.nn as nn
from torch.nn import LayerNorm
import pytorch_lightning as pl

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        
        return torch.matmul(weights, V)

class LSTMWithAttention(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = AttentionLayer(hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out = self.attention(lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)
        
        # Use last output
        last_out = attn_out[:, -1, :]
        
        # Predict
        output = self.fc(last_out)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

---

## Consequences

✅ Easy to implement custom architectures  
✅ Excellent for LSTM + attention  
✅ Strong community support  
✅ Better deployment options  

---

**Status:** ✅ ACCEPTED  
**Implementation:** Phase 4 (Model Development)
