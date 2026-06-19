#pragma once

#include <cstddef>
#include <unordered_map>
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

    double learning_rate() const { return learning_rate_; }
    void set_learning_rate(double learning_rate) { learning_rate_ = learning_rate; }
};

class SGD : public Optimizer {
private:
    double momentum_;
    double weight_decay_;
    std::unordered_map<Tensor*, std::vector<double>> velocity_;

public:
    SGD(const std::vector<Tensor*>& params,
        double learning_rate,
        double momentum = 0.0,
        double weight_decay = 0.0)
        : Optimizer(params, learning_rate),
          momentum_(momentum),
          weight_decay_(weight_decay) {}

    void step() override;
};

class Adam : public Optimizer {
private:
    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    std::size_t timestep_;

    std::unordered_map<Tensor*, std::vector<double>> first_moment_;
    std::unordered_map<Tensor*, std::vector<double>> second_moment_;

public:
    Adam(const std::vector<Tensor*>& params,
         double learning_rate = 0.001,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double epsilon = 1e-8,
         double weight_decay = 0.0)
        : Optimizer(params, learning_rate),
          beta1_(beta1),
          beta2_(beta2),
          epsilon_(epsilon),
          weight_decay_(weight_decay),
          timestep_(0) {}

    void step() override;
};
