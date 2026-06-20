#pragma once

#include <cstddef>
#include <vector>

#include "layer.h"
#include "tensor.h"

class Linear : public Layer {
private:
  std::size_t in_features_;
  std::size_t out_features_;

  Tensor weights_;
  Tensor bias_;

  Tensor grad_weights_;
  Tensor grad_bias_;

  Tensor input_cache_;

public:
  Linear(std::size_t in_features, std::size_t out_features);

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::vector<Tensor *> parameters() override;

  Tensor &weights();
  Tensor &bias();

  const Tensor &weights() const;
  const Tensor &bias() const;

  const Tensor &grad_weights() const;
  const Tensor &grad_bias() const;

  std::size_t in_features() const;
  std::size_t out_features() const;
};
