#pragma once

#include "common/Types.hpp"
#include <array>
#include <vector>
#include <optional>

namespace stac::model {

class ActionSpace {

    public:

        static ActionIndex encode(Square from, Plane plane);

        static std::pair<Square, Plane> decode(ActionIndex action);

        static std::optional<Square> get_target_square(
            Square from,
            Plane plane,
            Color side_to_move = Color::WHITE
        );

        static std::optional<Plane> get_plane(
            Square from,
            Square to,
            PieceType piece = PieceType::QUEEN,
            PieceType promotion = PieceType::QUEEN
        );

        static bool is_queen_move(Plane plane) {

            return plane < 56;

        }

        static bool is_knight_move(Plane plane) {

            return plane >= 56 && plane < 64;
            
        }

        static bool is_underpromotion(Plane plane) {

            return plane >= 64;

        }

        static PieceType get_promotion_piece(Plane plane);

        static std::vector<Square> get_valid_target(
            Square from,
            PieceType piece
        );

        static std::string describe_action(ActionIndex action);

        static bool is_valid_action(ActionIndex action) {

            return action >= 0 && action < constants::ACTION_SPACE_SIZE;

        }

    private:

        static constexpr std::array<int, 8> DIRECTION_FILES = {0, 0, 1, -1, 1, -1, 1, -1};

        static constexpr std::array<int, 8> DIRECTION_RANKS = {1, -1, 0, 0, 1, 1, -1, -1};

        static constexpr std::array<int, 8> KNIGHT_FILES = {1, 2, 2, 1, -1, -2, -2, -1};

        static constexpr std::array<int, 8> KNIGHT_RANKS = {2, 1, -1, -2, -2, -1, 1, 2};

        static bool is_on_board(int file, int rank);

        static Square to_square(int file, int rank);

        static std::pair<int, int> from_square(Square sq);

};


namespace action_mask_ops {

    void apply_mask(PolicyLogits& logits, const ActionMask& mask);

    int count_legal_actions(const ActionMask& mask);

    std::vector<ActionIndex> get_legal_actioin_indices(const ActionMask& mask);

    ActionMask all_legal_mask();

    ActionMask no_legal_mask();

    ActionMask combine_masks(const ActionMask& mask1, const ActionMask& mask2);



}

class ActionSampler {

    public:

        static ActionIndex sample(
            const PolicyLogits& logits,
            const ActionMask& mask,
            float temperature = 1.0f
        );

        static ActionIndex best_action(
            const PolicyLogits& logits,
            const ActionMask& mask
        );

        static std::vector<std::pair<ActionIndex, float>> top_k_actions(
            const PolicyLogits& logits,
            const ActionMask& mask,
            float epsilon = 0.25f
        );

        static void add_dirichlet_noise(
            PolicyLogits& logits,
            float alpha = 0.3f,
            float epsilon = 0.25f
        );
};

}