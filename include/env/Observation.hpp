#pragma once

#include "common/Types.hpp"

namespace stac {

class Observation {
public:
    Observation() = default;
    explicit Observation(const ObservationTensor& obs) : data(obs) {}
    
    const ObservationTensor& get_data() const { return data; }
    void set_data(const ObservationTensor& obs) { data = obs; }
    
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    
private:
    ObservationTensor data;
};

} // namespace stac
