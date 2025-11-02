#!/usr/bin/env python3
"""
Python Chess Adapter for STAC-RL

This provides a simple interface to chess game logic using python-chess library.
Can be used for:
1. Local CPU self-play
2. Testing the model
3. Interactive play

Install: pip3 install python-chess numpy
"""

import chess
import numpy as np
from typing import Tuple, List, Optional


class ChessAdapter:
    """Adapter for chess game logic using python-chess"""
    
    def __init__(self):
        self.board = chess.Board()
    
    def reset(self, fen: str = chess.STARTING_FEN) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset to starting position or custom FEN
        
        Returns:
            observation: (18, 8, 8) numpy array
            action_mask: (4672,) numpy array of legal actions
        """
        self.board = chess.Board(fen)
        return self.get_observation(), self.get_action_mask()
    
    def get_observation(self) -> np.ndarray:
        """
        Convert board to observation tensor
        
        Returns:
            (18, 8, 8) tensor with:
            - Planes 0-5: My pieces (P, N, B, R, Q, K)
            - Planes 6-11: Opponent pieces
            - Plane 12: Repetition count
            - Plane 13: Color (1=white to move, 0=black)
            - Plane 14: Total move count / 100
            - Plane 15: My castling rights
            - Plane 16: Opponent castling rights
            - Plane 17: No-progress count / 100
        """
        obs = np.zeros((18, 8, 8), dtype=np.float32)
        
        # Current player's pieces (planes 0-5)
        # Opponent's pieces (planes 6-11)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
            
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            # Piece type: PAWN=1, KNIGHT=2, etc. -> 0-indexed
            plane_offset = piece.piece_type - 1
            
            # If piece is opponent's, add 6 to plane index
            if piece.color != self.board.turn:
                plane_offset += 6
            
            obs[plane_offset, rank, file] = 1.0
        
        # Plane 12: Repetition count (simplified)
        # TODO: Track actual repetitions
        obs[12, :, :] = 0.0
        
        # Plane 13: Color (1=white, 0=black)
        obs[13, :, :] = 1.0 if self.board.turn == chess.WHITE else 0.0
        
        # Plane 14: Move count (normalized)
        obs[14, :, :] = min(self.board.fullmove_number / 100.0, 1.0)
        
        # Plane 15: My castling rights
        my_color = self.board.turn
        kingside = self.board.has_kingside_castling_rights(my_color)
        queenside = self.board.has_queenside_castling_rights(my_color)
        obs[15, :, 4:] = 1.0 if kingside else 0.0
        obs[15, :, :4] = 1.0 if queenside else 0.0
        
        # Plane 16: Opponent castling rights
        opp_color = not my_color
        kingside = self.board.has_kingside_castling_rights(opp_color)
        queenside = self.board.has_queenside_castling_rights(opp_color)
        obs[16, :, 4:] = 1.0 if kingside else 0.0
        obs[16, :, :4] = 1.0 if queenside else 0.0
        
        # Plane 17: No-progress count (halfmove clock)
        obs[17, :, :] = min(self.board.halfmove_clock / 100.0, 1.0)
        
        return obs
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of legal actions
        
        Returns:
            (4672,) binary mask where 1=legal, 0=illegal
        """
        mask = np.zeros(4672, dtype=np.uint8)
        
        for move in self.board.legal_moves:
            action_idx = self.uci_move_to_action_index(move)
            if action_idx is not None:
                mask[action_idx] = 1
        
        return mask
    
    def step(self, action_index: int) -> Tuple[np.ndarray, np.ndarray, float, bool, str]:
        """
        Apply action to board
        
        Args:
            action_index: Action index (0-4671)
        
        Returns:
            observation: Next state
            action_mask: Legal actions in next state
            reward: Reward (0 for ongoing, +1/-1 for win/loss, 0 for draw)
            done: Whether game is over
            info: Result string ("1-0", "0-1", "1/2-1/2", or "")
        """
        # Convert action index to UCI move
        move = self.action_index_to_uci_move(action_index)
        
        if move is None or move not in self.board.legal_moves:
            raise ValueError(f"Illegal action: {action_index}")
        
        # Apply move
        self.board.push(move)
        
        # Check if game is over
        done = self.board.is_game_over()
        reward = 0.0
        info = ""
        
        if done:
            result = self.board.result()
            info = result
            
            # Reward is from perspective of player who just moved
            if result == "1-0":  # White wins
                reward = 1.0 if not self.board.turn else -1.0
            elif result == "0-1":  # Black wins
                reward = -1.0 if not self.board.turn else 1.0
            else:  # Draw
                reward = 0.0
        
        obs = self.get_observation()
        mask = self.get_action_mask()
        
        return obs, mask, reward, done, info
    
    def uci_move_to_action_index(self, move: chess.Move) -> Optional[int]:
        """
        Convert UCI move to action index (0-4671)
        
        Action encoding: from_square (0-63) × 73 planes
        - Planes 0-55: Queen moves (8 directions × 7 distances)
        - Planes 56-63: Knight moves (8 L-shapes)
        - Planes 64-72: Underpromotions (3 directions × 3 pieces)
        """
        from_sq = move.from_square
        to_sq = move.to_square
        
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)
        to_rank = chess.square_rank(to_sq)
        to_file = chess.square_file(to_sq)
        
        # Compute direction
        delta_rank = to_rank - from_rank
        delta_file = to_file - from_file
        
        # Knight moves
        if abs(delta_rank) == 2 and abs(delta_file) == 1:
            knight_moves = [(2,1), (2,-1), (1,2), (1,-2), 
                          (-1,2), (-1,-2), (-2,1), (-2,-1)]
            try:
                plane = 56 + knight_moves.index((delta_rank, delta_file))
            except ValueError:
                return None
        elif abs(delta_rank) == 1 and abs(delta_file) == 2:
            knight_moves = [(2,1), (2,-1), (1,2), (1,-2), 
                          (-1,2), (-1,-2), (-2,1), (-2,-1)]
            try:
                plane = 56 + knight_moves.index((delta_rank, delta_file))
            except ValueError:
                return None
        
        # Underpromotions (pawn to rook, bishop, knight)
        elif move.promotion and move.promotion != chess.QUEEN:
            # Simplified: Use planes 64-72
            promo_pieces = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
            try:
                promo_idx = promo_pieces.index(move.promotion)
            except ValueError:
                return None
            
            # Direction: forward-left, forward, forward-right
            if delta_file == -1:
                dir_idx = 0
            elif delta_file == 0:
                dir_idx = 1
            elif delta_file == 1:
                dir_idx = 2
            else:
                return None
            
            plane = 64 + promo_idx * 3 + dir_idx
        
        # Queen moves (includes queen promotions)
        else:
            # Determine direction
            if delta_rank != 0 and delta_file != 0:
                # Diagonal
                if abs(delta_rank) != abs(delta_file):
                    return None
                
                if delta_rank > 0 and delta_file > 0:
                    direction = 4  # NE
                elif delta_rank > 0 and delta_file < 0:
                    direction = 5  # NW
                elif delta_rank < 0 and delta_file > 0:
                    direction = 6  # SE
                else:
                    direction = 7  # SW
                
                distance = abs(delta_rank) - 1
            elif delta_rank != 0:
                # Vertical
                direction = 0 if delta_rank > 0 else 1  # N or S
                distance = abs(delta_rank) - 1
            elif delta_file != 0:
                # Horizontal
                direction = 2 if delta_file > 0 else 3  # E or W
                distance = abs(delta_file) - 1
            else:
                return None
            
            if distance < 0 or distance > 6:
                return None
            
            plane = direction * 7 + distance
        
        action_index = from_sq * 73 + plane
        return action_index if 0 <= action_index < 4672 else None
    
    def action_index_to_uci_move(self, action_index: int) -> Optional[chess.Move]:
        """Convert action index back to UCI move (requires current board state)"""
        if not (0 <= action_index < 4672):
            return None
        
        from_sq = action_index // 73
        plane = action_index % 73
        
        # This is complex - for now, we validate against legal moves
        # In practice, you'd reverse the encoding logic
        
        # Quick approach: Try to find matching legal move
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)
        
        # Decode plane to get target square
        # (Full implementation would reverse the encoding logic above)
        # For now, search through legal moves
        
        for move in self.board.legal_moves:
            if self.uci_move_to_action_index(move) == action_index:
                return move
        
        return None
    
    def get_result(self) -> Optional[str]:
        """Get game result: '1-0', '0-1', '1/2-1/2', or None"""
        if self.board.is_game_over():
            return self.board.result()
        return None
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.board.is_game_over()
    
    def get_fen(self) -> str:
        """Get current position as FEN string"""
        return self.board.fen()
    
    def get_legal_moves_uci(self) -> List[str]:
        """Get list of legal moves in UCI format"""
        return [move.uci() for move in self.board.legal_moves]


if __name__ == "__main__":
    # Test the adapter
    print("Testing ChessAdapter...")
    
    adapter = ChessAdapter()
    obs, mask = adapter.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Legal moves: {mask.sum()}")
    print(f"FEN: {adapter.get_fen()}")
    print(f"Legal UCI moves: {adapter.get_legal_moves_uci()[:5]}...")  # First 5
    
    # Try a move (e2e4)
    e2e4 = chess.Move.from_uci("e2e4")
    action_idx = adapter.uci_move_to_action_index(e2e4)
    print(f"\nAction index for e2e4: {action_idx}")
    
    obs, mask, reward, done, info = adapter.step(action_idx)
    print(f"After e2e4:")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Legal moves: {mask.sum()}")
    print(f"  FEN: {adapter.get_fen()}")
    
    print("\nAdapter test passed!")
