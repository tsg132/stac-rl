#include "model/STACFlashModel.hpp"
#include <iostream>

namespace stac {

STACModel::STACModel(int observationDim, int actionDim, int modelDim, int numHeads, int numLayers)
    : transformer(observationDim, modelDim, numHeads, numLayers),
      actionSpace(actionDim),
      observationDim(observationDim),
      actionDim(actionDim) {
}

ActionVector STACModel::selectAction(const Observation& obs) {
    StateVector encoded = transformer.forward(obs.getState());
    return actionSpace.sample();
}

Float STACModel::evaluateValue(const Observation& obs) {
    StateVector encoded = transformer.forward(obs.getState());
    return 0.0f;
}

void STACModel::save(const std::string& path) const {
    std::cout << "Saving model to: " << path << std::endl;
}

void STACModel::load(const std::string& path) {
    std::cout << "Loading model from: " << path << std::endl;
}

} // namespace stac
