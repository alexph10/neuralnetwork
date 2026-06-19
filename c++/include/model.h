#pragma once

#include <memory>
#include <vector>

#include "layer.h"
#include "tensor.h"

class Model {
private:
    std::vector<std::shared_ptr<Layer>> layers_;

public:
    Model();

    void add(const std::shared_ptr<Layer>& layer);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    std::vector<Tensor*> parameters();

    std::size_t size() const;
};
