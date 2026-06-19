#include "model.h"

Model::Model() = default;

void Model::add(const std::shared_ptr<Layer>& layer) {
    layers_.push_back(layer);
}

Tensor Model::forward(const Tensor& input) {
    Tensor output = input;

    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }

    return output;
}

Tensor Model::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;

    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }

    return grad;
}

std::vector<Tensor*> Model::parameters() {
    std::vector<Tensor*> params;

    for (const auto& layer : layers_) {
        std::vector<Tensor*> layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}

std::size_t Model::size() const {
    return layers_.size();
}
