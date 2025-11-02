#pragma once

#include "common/Types.hpp"
#include "env/Observation.hpp"
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <iostream>
#include <map>
#include <string>
#include <string_view>


namespace stac::env {

// ============================================================================
// Base Adapter Interface
// ============================================================================

/**
 * Abstract interface for external chess engines/servers.
 * Adapters handle:
 * 1. FEN parsing and observation encoding
 * 2. Legal move generation
 * 3. Move execution and state updates
 */
class ChessAdapter {
public:
    virtual ~ChessAdapter() = default;
    
    // -------------------------------------------------------------------------
    // Core Methods
    // -------------------------------------------------------------------------
    
    /**
     * Parse FEN and create observation tensor.
     */
    virtual Observation fen_to_observation(const std::string& fen) const = 0;
    
    /**
     * Convert UCI move list to action mask.
     */
    virtual ActionMask uci_moves_to_mask(
        const std::vector<std::string>& uci_moves,
        Color side_to_move) const = 0;
    
    /**
     * Convert action index to UCI notation.
     */
    virtual std::string action_to_uci(
        ActionIndex action,
        const std::string& fen) const = 0;
    
    /**
     * Apply move and return new FEN.
     */
    virtual std::string apply_move(
        const std::string& fen,
        const std::string& uci_move) = 0;
    
    /**
     * Get legal moves for position.
     */
    virtual std::vector<std::string> get_legal_moves(
        const std::string& fen) = 0;
    
    /**
     * Check if game is over and get result.
     */
    virtual std::pair<bool, GameResult> is_game_over(
        const std::string& fen) = 0;
    
    // -------------------------------------------------------------------------
    // Optional Methods
    // -------------------------------------------------------------------------
    
    /**
     * Evaluate position (for reward shaping, optional).
     */
    virtual float evaluate_position(const std::string& fen) {
        return 0.0f;
    }
    
    /**
     * Get opening book move (optional).
     */
    virtual std::optional<std::string> get_book_move(const std::string& fen) {
        return std::nullopt;
    }
};

// ============================================================================
// Lichess Adapter
// ============================================================================

/**
 * Adapter for Lichess Bot API or analysis board.
 */
class LichessAdapter : public ChessAdapter {
public:
    struct Config {
        std::string api_token;
        std::string bot_name;
        int connection_timeout_ms = 5000;
        int move_timeout_ms = 1000;
        bool use_tablebase = false;
        bool use_opening_book = true;
    };
    
    explicit LichessAdapter(const Config& config);
    ~LichessAdapter() override;
    
    // Implement base interface
    Observation fen_to_observation(const std::string& fen) const override;
    ActionMask uci_moves_to_mask(
        const std::vector<std::string>& uci_moves,
        Color side_to_move) const override;
    std::string action_to_uci(
        ActionIndex action,
        const std::string& fen) const override;
    std::string apply_move(
        const std::string& fen,
        const std::string& uci_move) override;
    std::vector<std::string> get_legal_moves(const std::string& fen) override;
    std::pair<bool, GameResult> is_game_over(const std::string& fen) override;
    
    // Lichess-specific methods
    void start_game(const std::string& game_id);
    void resign_game();
    void offer_draw();
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// ============================================================================
// UCI Engine Adapter
// ============================================================================

/**
 * Adapter for UCI chess engines (Stockfish, Leela, etc.).
 */
class UCIAdapter : public ChessAdapter {
public:
    struct Config {
        std::string engine_path;
        std::map<std::string, std::string> engine_options;
        int threads = 1;
        int hash_mb = 16;
        bool use_nnue = true;
    };
    
    explicit UCIAdapter(const Config& config);
    ~UCIAdapter() override;
    
    // Implement base interface
    Observation fen_to_observation(const std::string& fen) const override;
    ActionMask uci_moves_to_mask(
        const std::vector<std::string>& uci_moves,
        Color side_to_move) const override;
    std::string action_to_uci(
        ActionIndex action,
        const std::string& fen) const override;
    std::string apply_move(
        const std::string& fen,
        const std::string& uci_move) override;
    std::vector<std::string> get_legal_moves(const std::string& fen) override;
    std::pair<bool, GameResult> is_game_over(const std::string& fen) override;
    float evaluate_position(const std::string& fen) override;
    
    // UCI-specific methods
    void set_position(const std::string& fen);
    std::string best_move(int depth = 10, int time_ms = 1000);
    void stop_search();
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// ============================================================================
// Adapter Utilities
// ============================================================================

namespace adapter_utils {
    
    /**
     * Parse UCI move string (e.g., "e2e4", "e7e8q").
     */
    struct UCIMove {
        Square from;
        Square to;
        char promotion = '\0';  // 'q', 'r', 'b', 'n', or '\0'
        
        static UCIMove parse(const std::string& uci);
        std::string to_string() const;
    };
    
    /**
     * Convert between coordinate systems.
     */
    Square algebraic_to_square(const std::string& algebraic);  // "e4" -> 28
    std::string square_to_algebraic(Square square);  // 28 -> "e4"
    
    /**
     * Map UCI move to action plane.
     */
    Plane uci_to_plane(const UCIMove& move, Color side_to_move);
    
    /**
     * Map action to target square.
     */
    Square plane_to_target_square(Square from, Plane plane, Color side_to_move);
    
} // namespace adapter_utils

// ============================================================================
// Adapter Factory
// ============================================================================

class AdapterFactory {
public:
    static std::unique_ptr<ChessAdapter> create_lichess(
        const LichessAdapter::Config& config);
    
    static std::unique_ptr<ChessAdapter> create_uci(
        const UCIAdapter::Config& config);
    
    static std::unique_ptr<ChessAdapter> create_from_config(
        const std::string& type,
        const std::string& config_json);
};

} // namespace stac::env