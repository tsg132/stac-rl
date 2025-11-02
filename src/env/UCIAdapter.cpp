/**
 * UCI Chess Engine Adapter Implementation
 * 
 * Connects to any UCI-compatible chess engine (Stockfish, Leela, etc.)
 * Handles FEN parsing, move generation, and game state management.
 */

#include "env/Adapter.hpp"
#include <sstream>
#include <iostream>
#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace stac::env {

// ============================================================================
// UCIAdapter Implementation
// ============================================================================

class UCIAdapter::Impl {
public:
    explicit Impl(const UCIAdapter::Config& config) 
        : config_(config), engine_ready_(false) {
        start_engine();
    }
    
    ~Impl() {
        stop_engine();
    }
    
    void start_engine() {
        // TODO: Implement process spawning and UCI communication
        // For now, we'll use a simple internal chess logic
        std::cout << "UCI Engine adapter created (path: " << config_.engine_path << ")" << std::endl;
        engine_ready_ = true;
    }
    
    void stop_engine() {
        if (engine_ready_) {
            // TODO: Send "quit" command to engine
            engine_ready_ = false;
        }
    }
    
    std::vector<std::string> get_legal_moves(const std::string& fen) {
        // TODO: Send "position fen ... \ngo perft 1" and parse output
        // For now, return placeholder
        return {"e2e4", "d2d4", "g1f3", "b1c3"};
    }
    
    std::string apply_move(const std::string& fen, const std::string& uci_move) {
        // TODO: Send position and move, get resulting FEN
        // For now, return original FEN (placeholder)
        return fen;
    }
    
    bool is_game_over(const std::string& fen) {
        // TODO: Check for checkmate, stalemate, etc.
        return false;
    }
    
    float evaluate(const std::string& fen) {
        // TODO: Send "go depth 10" and parse score
        return 0.0f;
    }

private:
    UCIAdapter::Config config_;
    bool engine_ready_;
    
    // TODO: Add process handles for engine communication
};

// ============================================================================
// UCIAdapter Public Interface
// ============================================================================

UCIAdapter::UCIAdapter(const Config& config) 
    : pimpl_(std::make_unique<Impl>(config)) {
}

UCIAdapter::~UCIAdapter() = default;

Observation UCIAdapter::fen_to_observation(const std::string& fen) const {
    // Parse FEN and create 18x8x8 observation tensor
    Observation obs;
    obs.planes.resize(18 * 64, 0.0f);
    
    // TODO: Implement FEN parsing
    // For now, return empty observation
    
    return obs;
}

ActionMask UCIAdapter::uci_moves_to_mask(
    const std::vector<std::string>& uci_moves,
    Color side_to_move) const {
    
    ActionMask mask;
    mask.legal_actions.resize(4672, 0);
    
    // TODO: Convert UCI moves to action indices
    // For now, mark first few actions as legal
    for (size_t i = 0; i < std::min(uci_moves.size(), size_t(100)); ++i) {
        mask.legal_actions[i] = 1;
    }
    
    return mask;
}

std::string UCIAdapter::action_to_uci(
    ActionIndex action,
    const std::string& fen) const {
    
    // TODO: Implement action index to UCI conversion
    // This requires understanding the action encoding
    return "e2e4";  // Placeholder
}

std::string UCIAdapter::apply_move(
    const std::string& fen,
    const std::string& uci_move) {
    
    return pimpl_->apply_move(fen, uci_move);
}

std::vector<std::string> UCIAdapter::get_legal_moves(const std::string& fen) {
    return pimpl_->get_legal_moves(fen);
}

std::pair<bool, GameResult> UCIAdapter::is_game_over(const std::string& fen) {
    bool done = pimpl_->is_game_over(fen);
    GameResult result = GameResult::ONGOING;
    
    // TODO: Determine actual result
    
    return {done, result};
}

float UCIAdapter::evaluate_position(const std::string& fen) {
    return pimpl_->evaluate(fen);
}

void UCIAdapter::set_position(const std::string& fen) {
    // TODO: Send position to engine
}

std::string UCIAdapter::best_move(int depth, int time_ms) {
    // TODO: Ask engine for best move
    return "e2e4";
}

void UCIAdapter::stop_search() {
    // TODO: Send "stop" command
}

// ============================================================================
// Lichess Adapter (Stub)
// ============================================================================

class LichessAdapter::Impl {
public:
    explicit Impl(const LichessAdapter::Config& config) {
        std::cout << "Lichess adapter created (bot: " << config.bot_name << ")" << std::endl;
    }
};

LichessAdapter::LichessAdapter(const Config& config)
    : pimpl_(std::make_unique<Impl>(config)) {
}

LichessAdapter::~LichessAdapter() = default;

Observation LichessAdapter::fen_to_observation(const std::string& fen) const {
    Observation obs;
    obs.planes.resize(18 * 64, 0.0f);
    return obs;
}

ActionMask LichessAdapter::uci_moves_to_mask(
    const std::vector<std::string>& uci_moves,
    Color side_to_move) const {
    
    ActionMask mask;
    mask.legal_actions.resize(4672, 0);
    return mask;
}

std::string LichessAdapter::action_to_uci(
    ActionIndex action,
    const std::string& fen) const {
    return "e2e4";
}

std::string LichessAdapter::apply_move(
    const std::string& fen,
    const std::string& uci_move) {
    return fen;
}

std::vector<std::string> LichessAdapter::get_legal_moves(const std::string& fen) {
    return {"e2e4"};
}

std::pair<bool, GameResult> LichessAdapter::is_game_over(const std::string& fen) {
    return {false, GameResult::ONGOING};
}

void LichessAdapter::start_game(const std::string& game_id) {
    // TODO: Connect to Lichess game
}

void LichessAdapter::resign_game() {
    // TODO: Send resign
}

void LichessAdapter::offer_draw() {
    // TODO: Send draw offer
}

// ============================================================================
// Adapter Factory
// ============================================================================

std::unique_ptr<ChessAdapter> AdapterFactory::create_uci(
    const UCIAdapter::Config& config) {
    return std::make_unique<UCIAdapter>(config);
}

std::unique_ptr<ChessAdapter> AdapterFactory::create_lichess(
    const LichessAdapter::Config& config) {
    return std::make_unique<LichessAdapter>(config);
}

std::unique_ptr<ChessAdapter> AdapterFactory::create_from_config(
    const std::string& type,
    const std::string& config_json) {
    
    // TODO: Parse JSON config
    
    if (type == "uci") {
        UCIAdapter::Config config;
        config.engine_path = "/usr/local/bin/stockfish";
        return create_uci(config);
    } else if (type == "lichess") {
        LichessAdapter::Config config;
        return create_lichess(config);
    }
    
    throw std::runtime_error("Unknown adapter type: " + type);
}

// ============================================================================
// Adapter Utilities
// ============================================================================

namespace adapter_utils {

UCIMove UCIMove::parse(const std::string& uci) {
    if (uci.length() < 4) {
        throw std::invalid_argument("Invalid UCI move: " + uci);
    }
    
    UCIMove move;
    move.from = algebraic_to_square(uci.substr(0, 2));
    move.to = algebraic_to_square(uci.substr(2, 2));
    
    if (uci.length() >= 5) {
        move.promotion = uci[4];
    }
    
    return move;
}

std::string UCIMove::to_string() const {
    std::string s = square_to_algebraic(from) + square_to_algebraic(to);
    if (promotion != '\0') {
        s += promotion;
    }
    return s;
}

Square algebraic_to_square(const std::string& algebraic) {
    if (algebraic.length() != 2) {
        return 0;
    }
    
    int file = algebraic[0] - 'a';  // 0-7
    int rank = algebraic[1] - '1';  // 0-7
    
    if (file < 0 || file > 7 || rank < 0 || rank > 7) {
        return 0;
    }
    
    return rank * 8 + file;  // 0-63
}

std::string square_to_algebraic(Square square) {
    int file = square % 8;
    int rank = square / 8;
    
    std::string s;
    s += char('a' + file);
    s += char('1' + rank);
    
    return s;
}

Plane uci_to_plane(const UCIMove& move, Color side_to_move) {
    // TODO: Implement move to plane encoding
    // This is complex - needs direction and distance calculation
    return 0;
}

Square plane_to_target_square(Square from, Plane plane, Color side_to_move) {
    // TODO: Implement plane to target square
    return 0;
}

} // namespace adapter_utils

} // namespace stac::env
