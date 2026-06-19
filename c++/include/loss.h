#pragma once

#include <cstddef>
#include <vector>

#include "tensor.h"

class Loss {
public:
    virtual ~Loss() = default;

    virtual double forward(const Tensor& prediction, const Tensor& target) = 0;
    virtual Tensor backward() = 0;
};

class MSELoss : public Loss {
private:
    Tensor prediction_;
    Tensor target_;
    double last_loss_ = 0.0;

public:
    MSELoss() = default;

    double forward(const Tensor& prediction, const Tensor& target) override;
    Tensor backward() override;

    double value() const { return last_loss_; }
};

class CrossEntropyLoss : public Loss {
private:
    Tensor logits_;
    Tensor target_;
    Tensor probabilities_;
    double last_loss_ = 0.0;

public:
    CrossEntropyLoss() = default;

    double forward(const Tensor& prediction, const Tensor& target) override;
    Tensor backward() override;

    double value() const { return last_loss_; }
};
