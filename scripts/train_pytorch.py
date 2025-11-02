#!/usr/bin/env python3
"""
Simple PyTorch Training Script for STAC-RL

Train a chess model on CPU (or GPU if available).
This implements the same architecture as the C++ code but in PyTorch.

Install:
    pip3 install torch numpy python-chess

Usage:
    # Generate data first
    python3 scripts/self_play.py --num-games 100 --output data/games.npz
    
    # Train on CPU
    python3 scripts/train_pytorch.py --data data/games.npz --device cpu
    
    # Or GPU if available
    python3 scripts/train_pytorch.py --data data/games.npz --device cuda
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time


class FlashTransformerBlock(nn.Module):
    """Single transformer block with multi-head attention"""
    
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class STACModel(nn.Module):
    """STAC-RL Model - Transformer for Chess"""
    
    def __init__(self, 
                 d_model=512,
                 n_heads=8,
                 n_layers=6,
                 dropout=0.1,
                 num_actions=4672):
        super().__init__()
        
        # Input embedding: 18 channels → d_model
        # Each square is a position, 64 total
        self.input_embed = nn.Sequential(
            nn.Conv2d(18, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
        )
        
        # Positional encoding for 64 squares
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            FlashTransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Policy head (outputs action logits)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_actions)
        )
        
        # Value head (outputs position evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, obs):
        """
        Args:
            obs: (batch, 18, 8, 8) board observation
        
        Returns:
            policy_logits: (batch, 4672) action logits
            value: (batch, 1) position value
        """
        batch_size = obs.shape[0]
        
        # Input embedding: (batch, 18, 8, 8) → (batch, d_model, 8, 8)
        x = self.input_embed(obs)
        
        # Reshape to sequence: (batch, d_model, 8, 8) → (batch, 64, d_model)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Pool across positions (mean pooling)
        x_pooled = x.mean(dim=1)  # (batch, d_model)
        
        # Policy and value heads
        policy_logits = self.policy_head(x_pooled)
        value = self.value_head(x_pooled)
        
        return policy_logits, value


def compute_loss(policy_logits, value, actions, rewards, action_masks):
    """
    Compute policy and value loss
    
    Args:
        policy_logits: (batch, 4672) model output
        value: (batch, 1) value predictions
        actions: (batch,) actions taken
        rewards: (batch,) actual rewards
        action_masks: (batch, 4672) legal action masks
    """
    batch_size = policy_logits.shape[0]
    
    # Mask illegal actions
    masked_logits = policy_logits.clone()
    masked_logits[action_masks == 0] = -1e9
    
    # Policy loss (cross-entropy)
    log_probs = F.log_softmax(masked_logits, dim=1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    policy_loss = -action_log_probs.mean()
    
    # Value loss (MSE)
    value_loss = F.mse_loss(value.squeeze(1), rewards)
    
    # Total loss
    total_loss = policy_loss + 0.5 * value_loss
    
    return total_loss, policy_loss, value_loss


def load_data(data_path):
    """Load training data from .npz file"""
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    
    observations = torch.from_numpy(data['observations']).float()
    actions = torch.from_numpy(data['actions']).long()
    rewards = torch.from_numpy(data['rewards']).float()
    
    print(f"  Loaded {len(observations)} positions")
    print(f"  Observations shape: {observations.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    
    return observations, actions, rewards


def create_dummy_masks(batch_size, num_actions=4672):
    """Create dummy action masks (all legal for now)"""
    # TODO: Load actual action masks from games
    return torch.ones(batch_size, num_actions, dtype=torch.uint8)


def train(args):
    """Main training loop"""
    
    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    if device.type == 'cpu':
        print("NOTE: Training on CPU will be slower. Consider using GPU in Colab.")
    
    # Load data
    observations, actions, rewards = load_data(args.data)
    num_samples = len(observations)
    
    # Create model
    print(f"\nCreating model (d_model={args.d_model}, layers={args.n_layers})...")
    model = STACModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, args.batch_size):
            batch_indices = indices[i:i + args.batch_size]
            
            # Get batch
            batch_obs = observations[batch_indices].to(device)
            batch_actions = actions[batch_indices].to(device)
            batch_rewards = rewards[batch_indices].to(device)
            batch_masks = create_dummy_masks(len(batch_indices)).to(device)
            
            # Forward pass
            policy_logits, value = model(batch_obs)
            
            # Compute loss
            loss, policy_loss, value_loss = compute_loss(
                policy_logits, value, batch_actions, batch_rewards, batch_masks
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            num_batches += 1
        
        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Loss: {avg_loss:.4f} | Policy: {avg_policy_loss:.4f} | Value: {avg_value_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"model_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = Path(args.checkpoint_dir) / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train STAC-RL model with PyTorch')
    
    # Data
    parser.add_argument('--data', type=str, default='data/games.npz',
                       help='Path to training data')
    
    # Model architecture
    parser.add_argument('--d-model', type=int, default=256,
                       help='Model dimension (256=small, 512=base)')
    parser.add_argument('--n-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4,
                       help='Number of transformer layers (4=small, 6=base)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (reduce if OOM on CPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STAC-RL PyTorch Training")
    print("="*60)
    
    train(args)


if __name__ == "__main__":
    main()
