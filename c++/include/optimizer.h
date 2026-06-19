#pragma once

#include <vector>

#include "tensor.h"

class Optimizer {
protected:
    std::vector<Tensor*> params_;
    double learning_rate_;

public:
    Optimizer(const std::vector<Tensor*>& params, double learning_rate)
        : params_(params), learning_rate_(learning_rate) {}

    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad();
};

class SGD : public Optimizer {
public:
    SGD(const std::vector<Tensor*>& params, double learning_rate)
        : Optimizer(params, learning_rate) {}

    void step() override;
};
