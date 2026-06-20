#pragma once

#include <cstddef>
#include <vector>

class Shape {
private:
    std::vector<std::size_t> dims_;
    std::vector<std::size_t> strides_;

    void compute_strides();

public:
    Shape() = default;
    Shape(const std::vector<std::size_t>& dims);

    const std::vector<std::size_t>& dims() const;
    const std::vector<std::size_t>& strides() const;

    std::size_t rank() const;
    std::size_t numel() const;
    std::size_t dim(std::size_t axis) const;

    bool empty() const;

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;
};
