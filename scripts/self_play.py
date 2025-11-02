#!/usr/bin/env python3
"""
Self-Play Script for STAC-RL

Generates games by having the model play against itself.
Saves game data for training.

Usage:
    python scripts/self_play.py --num-games 100 --output data/games.npz
"""

import argparse
import numpy as np
from python_adapter import ChessAdapter
import sys
from pathlib import Path


def random_policy(obs, action_mask):
    """Random policy for testing - replace with your model"""
    legal_actions = np.where(action_mask == 1)[0]
    if len(legal_actions) == 0:
        raise ValueError("No legal moves!")
    return np.random.choice(legal_actions)


def softmax(x):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def model_policy(obs, action_mask):
    """
    Your model policy - replace this with actual model inference
    
    For now, uses temperature-scaled random policy
    """
    # TODO: Replace with actual model forward pass
    # logits, value = model.forward(obs)
    
    # Random logits for demonstration
    logits = np.random.randn(4672).astype(np.float32)
    
    # Mask illegal moves
    masked_logits = np.where(action_mask == 1, logits, -np.inf)
    
    # Sample from policy
    probs = softmax(masked_logits)
    action = np.random.choice(4672, p=probs)
    
    return action


def play_game(policy_fn, max_moves=512, verbose=False):
    """
    Play one game
    
    Returns:
        game_data: Dict with observations, actions, rewards, etc.
    """
    adapter = ChessAdapter()
    obs, mask = adapter.reset()
    
    observations = []
    actions = []
    rewards = []
    dones = []
    
    for move_num in range(max_moves):
        # Store observation
        observations.append(obs.copy())
        
        # Get action from policy
        action = policy_fn(obs, mask)
        actions.append(action)
        
        if verbose:
            uci_move = adapter.action_index_to_uci_move(action)
            print(f"Move {move_num + 1}: {uci_move.uci() if uci_move else 'INVALID'}")
        
        # Step environment
        try:
            obs, mask, reward, done, info = adapter.step(action)
        except ValueError as e:
            print(f"Illegal move attempted: {e}")
            # Game ends as loss
            reward = -1.0
            done = True
            info = "illegal"
        
        rewards.append(reward)
        dones.append(done)
        
        if done:
            if verbose:
                print(f"Game over: {info}")
                print(f"Total moves: {move_num + 1}")
            break
    
    # If max moves reached without ending, declare draw
    if not done:
        info = '1/2-1/2'
        if verbose:
            print(f"Game reached max moves ({max_moves}), declared draw")
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'result': info if done else info,
        'num_moves': len(observations)
    }


def main():
    parser = argparse.ArgumentParser(description='Self-play for STAC-RL')
    parser.add_argument('--num-games', type=int, default=10,
                       help='Number of games to play')
    parser.add_argument('--output', type=str, default='data/games.npz',
                       help='Output file for game data')
    parser.add_argument('--policy', type=str, default='random',
                       choices=['random', 'model'],
                       help='Policy to use')
    parser.add_argument('--max-moves', type=int, default=512,
                       help='Maximum moves per game')
    parser.add_argument('--verbose', action='store_true',
                       help='Print game moves')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select policy
    if args.policy == 'random':
        policy_fn = random_policy
    else:
        policy_fn = model_policy
    
    print(f"Playing {args.num_games} games with {args.policy} policy...")
    
    all_games = []
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, 'unfinished': 0, 'illegal': 0}
    
    for game_num in range(args.num_games):
        if args.verbose:
            print(f"\n{'='*50}")
            print(f"Game {game_num + 1}/{args.num_games}")
            print(f"{'='*50}")
        else:
            if (game_num + 1) % 10 == 0:
                print(f"Played {game_num + 1}/{args.num_games} games...")
        
        game_data = play_game(policy_fn, args.max_moves, args.verbose)
        all_games.append(game_data)
        
        result = game_data['result']
        results[result] = results.get(result, 0) + 1
        
        if not args.verbose:
            # Print progress summary
            sys.stdout.write(f"\rGames: {game_num + 1} | " +
                           f"W: {results['1-0']} | " +
                           f"B: {results['0-1']} | " +
                           f"D: {results['1/2-1/2']} | " +
                           f"Avg moves: {np.mean([g['num_moves'] for g in all_games]):.1f}")
            sys.stdout.flush()
    
    print("\n\nSelf-play complete!")
    print(f"\nResults:")
    print(f"  White wins: {results['1-0']} ({100*results['1-0']/args.num_games:.1f}%)")
    print(f"  Black wins: {results['0-1']} ({100*results['0-1']/args.num_games:.1f}%)")
    print(f"  Draws: {results['1/2-1/2']} ({100*results['1/2-1/2']/args.num_games:.1f}%)")
    print(f"  Average game length: {np.mean([g['num_moves'] for g in all_games]):.1f} moves")
    
    # Save all games
    print(f"\nSaving games to {args.output}...")
    
    # Concatenate all game data
    all_obs = np.concatenate([g['observations'] for g in all_games])
    all_actions = np.concatenate([g['actions'] for g in all_games])
    all_rewards = np.concatenate([g['rewards'] for g in all_games])
    
    # Save results as separate arrays (avoid pickle)
    np.savez_compressed(
        args.output,
        observations=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        num_games=args.num_games,
        white_wins=results['1-0'],
        black_wins=results['0-1'],
        draws=results['1/2-1/2']
    )
    
    print(f"Saved {len(all_obs)} positions from {args.num_games} games")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
