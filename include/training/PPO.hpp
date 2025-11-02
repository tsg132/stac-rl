#pragma once

#include "model/STACFlashModel.hpp"
#include "env/Environment.hpp"

namespace stac {

class PPO {
public:
    PPO(STACModel& model, Environment& env, Float learningRate, Float clipEpsilon);
    
    void train(int numEpisodes);
    Float computeAdvantage(Float reward, Float value, Float nextValue);
    
private:
    STACModel& model;
    Environment& env;
    Float learningRate;
    Float clipEpsilon;
};

} // namespace stac
