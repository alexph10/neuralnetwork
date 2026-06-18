#pragma once

#include <vector>
#include <cstddef>
#include <initializer_list>

class Tensor {
private:
  std::vector<float> data_;
  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;

  std::size_t compute_offset(const std::vector<std::size_t> &indices) const;
  void compute_strides();
  static std::size_t compute_numel(const std::vector<std::size_t> &shape);

public:
  Tensor();
  explicit Tensor(const std::vector<std::size_t> &shape);
  Tensor(const std::vector<std::size_t>& shape, float value);
  Tensor(const std::vector<std::size_t> &shape,
         const std::vector<float> &values);

  std::size_t rank() const;
  std::size_t numel() const;
  bool empty() const;

  const std::vector<std::size_t> &shape() const;
  const std::vector<std::size_t> &strides() const;
  const std::vector<float> &data() const;
  std::vector<float> &data();

  float &at(const std::vector<std::size_t> &indices);
  const float &at(const std::vector<std::size_t> &indices) const;

  float &operator[](std::size_t index);
  const float &operator[](std::size_t index) const;

  void reshape(const std::vector<std::size_t> &new_shape);
  void fill(float value);

  void print() const;

  static Tensor zeros(const std::vector<std::size_t> &shape);
  static Tensor ones(const std::vector<std::size_t> &shape);
};
