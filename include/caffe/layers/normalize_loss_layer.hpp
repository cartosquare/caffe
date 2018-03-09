#ifndef CAFFE_NORMALIZE_LOSS_LAYER_HPP_
#define CAFFE_NORMALIZE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Normalize loss @f$
 *          E = \frac{\sum{i}{\frac{d_k}{s_k} {\sigma(v_k = 1)}}}{\sum{i} {\sigma(v_k = 1)}}
 *
 * @param bottom input Blob vector (length 4)
 * @param top output Blob vector (length 1)
 *
 */
template <typename Dtype>
class NormalizeLossLayer : public LossLayer<Dtype> {
 public:
  explicit NormalizeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NormalizeLoss"; }
  /**
   * Unlike most loss layers, in the NormalizeLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  virtual inline int ExactNumBottomBlobs() const { return 4; } 

 protected:
  /// @copydoc NormalizeLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Normalize error gradient w.r.t. the inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int n_visiables_;
};

}  // namespace caffe

#endif  // CAFFE_NORMALIZE_LOSS_LAYER_HPP_
