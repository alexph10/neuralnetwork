#include "tensor.h"

#include <iostream>
#include <stdexcept>

std::size_t Tensor::compute_numel(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }

    std::size_t total = 1;
    for (std::size_t dim : shape) {
        total *= dim;
    }
    return total;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());

    if (shape_.empty()) {
        return;
    }

    strides_[shape_.size() - 1] = 1;

    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

std::size_t Tensor::compute_offset(const std::vector<std::size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index rank does not match tensor rank.");
    }

    std::size_t offset = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of bounds.");
        }
        offset += indices[i] * strides_[i];
    }

    return offset;
}

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<std::size_t>& shape)
    : shape_(shape), data_(compute_numel(shape), 0.0f) {
    compute_strides();
}

Tensor::Tensor(const std::vector<std::size_t>& shape, float value)
    : shape_(shape), data_(compute_numel(shape), value) {
    compute_strides();
}

Tensor::Tensor(const std::vector<std::size_t>& shape, const std::vector<float>& values)
    : shape_(shape), data_(values) {
    if (data_.size() != compute_numel(shape_)) {
        throw std::invalid_argument("Value count does not match tensor shape.");
    }
    compute_strides();
}

std::size_t Tensor::rank() const {
    return shape_.size();
}

std::size_t Tensor::numel() const {
    return data_.size();
}

bool Tensor::empty() const {
    return data_.empty();
}

const std::vector<std::size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<std::size_t>& Tensor::strides() const {
    return strides_;
}

const std::vector<float>& Tensor::data() const {
    return data_;
}

std::vector<float>& Tensor::data() {
    return data_;
}

float& Tensor::at(const std::vector<std::size_t>& indices) {
    return data_[compute_offset(indices)];
}

const float& Tensor::at(const std::vector<std::size_t>& indices) const {
    return data_[compute_offset(indices)];
}

float& Tensor::operator[](std::size_t index) {
    return data_.at(index);
}

const float& Tensor::operator[](std::size_t index) const {
    return data_.at(index);
}

void Tensor::reshape(const std::vector<std::size_t>& new_shape) {
    if (compute_numel(new_shape) != numel()) {
        throw std::invalid_argument("Reshape changes total number of elements.");
    }

    shape_ = new_shape;
    compute_strides();
}

void Tensor::fill(float value) {
    for (float& x : data_) {
        x = value;
    }
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "], data=[";
    for (std::size_t i = 0; i < data_.size(); ++i) {
        std::cout << data_[i];
        if (i + 1 < data_.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "])\n";
}

Tensor Tensor::zeros(const std::vector<std::size_t>& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const std::vector<std::size_t>& shape) {
    return Tensor(shape, 1.0f);
}
