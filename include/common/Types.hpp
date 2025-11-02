#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <string>

namespace stac {

using Square = int8_t;

using Plane = int8_t;

using ActionIndex = int32_t;

using Bitboard = uint64_t;

enum class Color : uint8_t {

    WHITE = 0,

    BLACK = 1

};

enum class PieceType : uint8_t {

    PAWN = 0,

    KNIGHT = 1,

    BISHOP = 2,

    ROOK = 3,

    QUEEN = 4,

    KING = 5

};

namespace constants {

    constexpr int BOARD_SIZE = 8;

    constexpr int NUM_SQUARES = 64;

    constexpr int NUM_PLANES = 73;

    constexpr int ACTION_SPACE_SIZE = NUM_SQUARES * NUM_PLANES;

    constexpr int PIECE_PLANES = 12;

    constexpr int META_PLANES = 6;

    constexpr int TOTAL_PLANES = PIECE_PLANES + META_PLANES;

    constexpr int OBS_SIZE = TOTAL_PLANES * NUM_SQUARES;

    constexpr int DEFAULT_EMBEDDING_DIM = 256;

    constexpr int DEFAULT_NUM_HEADS = 8;

    constexpr int DEFAULT_NUM_LAYERS = 8;

    constexpr int DEFAULT_MLP_HIDDEN = 1024;


}


struct CastlingRights {

    bool white_kingside = false;

    bool white_queenside = false;

    bool black_kingside = false;

    bool black_queenside = false;

    bool can_castle(Color color, bool kingside) const {

        if (color == Color::WHITE) {

            return kingside ? white_kingside : white_queenside;

        } else {

            return kingside ? black_kingside : black_queenside;

        }

    }
};


struct EnPassant {

    bool valid = false;

    Square file = -1;

};

struct Move {

    Square from;

    Square to;

    Plane plane;

    ActionIndex to_index() const {

        return from * constants::NUM_PLANES + plane;

    }

    static Move from_index(ActionIndex idx) {

        return Move {

            static_cast<Square>(idx / constants::NUM_PLANES),
            -1,
            static_cast<Plane>(idx % constants::NUM_PLANES)


        };
    }


};

using ObservationTensor = std::array<float, constants::OBS_SIZE>;

using ActionMask = std::array<uint8_t, constants::ACTION_SPACE_SIZE>;

using PolicyLogits = std::array<float, constants::ACTION_SPACE_SIZE>;

template<typename T>

using Batch = std::vector<T>;

enum class GameResult : int8_t {

    ONGOING = 0,

    WHITE_WIN = 1,

    BLACK_WIN = -1,

    DRAW = 0

};

struct StepResult {

    ObservationTensor observation;

    float reward;

    bool done;

    GameResult result;

    ActionMask legal_actions;

};

struct ModelOutput {

    PolicyLogits policy_logits;

    float value;

};


}