#include "linear.h"

#include <stdexcept>

#include "ops.h"

Linear::Linear(std::size_t in_features, std::size_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weights_(ops::zeros({in_features, out_features})),
      bias_(ops::zeros({1, out_features})),
      grad_weights_(ops::zeros({in_features, out_features})),
      grad_bias_(ops::zeros({1, out_features})) {
}

Tensor Linear::forward(const Tensor& input) {
    input_cache_ = input;

    Tensor output = ops::matmul(input, weights_);
    output = ops::add(output, bias_);

    return output;
}

Tensor Linear::backward(const Tensor& grad_output) {
    Tensor input_t = ops::transpose(input_cache_);
    Tensor weights_t = ops::transpose(weights_);

    grad_weights_ = ops::matmul(input_t, grad_output);
    grad_bias_ = grad_output;
    Tensor grad_input = ops::matmul(grad_output, weights_t);

    return grad_input;
}

std::vector<Tensor*> Linear::parameters() {
    return {&weights_, &bias_};
}

Tensor& Linear::weights() {
    return weights_;
}

Tensor& Linear::bias() {
    return bias_;
}

const Tensor& Linear::weights() const {
    return weights_;
}

const Tensor& Linear::bias() const {
    return bias_;
}

const Tensor& Linear::grad_weights() const {
    return grad_weights_;
}

const Tensor& Linear::grad_bias() const {
    return grad_bias_;
}

std::size_t Linear::in_features() const {
    return in_features_;
}

std::size_t Linear::out_features() const {
    return out_features_;
}
