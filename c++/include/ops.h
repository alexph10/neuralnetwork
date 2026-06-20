#pragma once

#include <cstddef>
#include <vector>

#include "tensor.h"

namespace ops {

// Tensor creation
Tensor zeros(const std::vector<std::size_t>& shape);
Tensor ones(const std::vector<std::size_t>& shape);
Tensor full(const std::vector<std::size_t>& shape, double value);

// Elementwise tensor-tensor operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);

// Elementwise tensor-scalar operations
Tensor add_scalar(const Tensor& a, double scalar);
Tensor subtract_scalar(const Tensor& a, double scalar);
Tensor multiply_scalar(const Tensor& a, double scalar);
Tensor divide_scalar(const Tensor& a, double scalar);

// Unary operations
Tensor negate(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor square(const Tensor& a);
Tensor clamp(const Tensor& a, double min_value, double max_value);

// Linear algebra
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& a);

// Shape operations
Tensor reshape(const Tensor& a, const std::vector<std::size_t>& new_shape);
Tensor flatten(const Tensor& a);

// Reductions
double sum(const Tensor& a);
double mean(const Tensor& a);
double max(const Tensor& a);
double min(const Tensor& a);

// Activations
Tensor relu(const Tensor& a);
Tensor relu_backward(const Tensor& grad_output, const Tensor& input);

Tensor sigmoid(const Tensor& a);
Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output);

Tensor tanh(const Tensor& a);
Tensor tanh_backward(const Tensor& grad_output, const Tensor& output);

Tensor softmax(const Tensor& a);

// Utilities
bool same_shape(const Tensor& a, const Tensor& b);
std::size_t numel(const Tensor& a);

} // namespace ops
