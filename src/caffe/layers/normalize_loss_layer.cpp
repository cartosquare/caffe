#include <vector>

#include "caffe/layers/normalize_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[1]->count(1), bottom[2]->count(1) * 2)
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[3]->count(1), 1) << "Last input must has dimension 1";

}

template <typename Dtype>
void NormalizeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  const Dtype* predict = bottom[0]->cpu_data();
  const Dtype* ground_truth = bottom[1]->cpu_data();
  const Dtype* visiable = bottom[2]->cpu_data();
  const Dtype* normalize_param = bottom[3]->cpu_data();

  int batch_size = bottom[0]->shape(0);
  int sample_size = bottom[0]->shape(1);
  int pts = sample_size / 2;

  Dtype total_error = 0.0;
  n_visiables_ = 0;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < pts; ++j) {
      Dtype p_x = predict[i * sample_size + j * 2];
      Dtype p_y = predict[i * sample_size + j * 2 + 1];

      Dtype g_x = ground_truth[i * sample_size + j * 2];
      Dtype g_y = ground_truth[i * sample_size + j * 2 + 1];

      Dtype v = visiable[i * pts + j];
      if (v <= 0.5) {
        // this keypoint is invisiable
        continue;
      }
      // calculate normalized distance
      Dtype dist = sqrtf((p_x - g_x) * (p_x - g_x) + (p_y - g_y) * (p_y - g_y));
      dist /= normalize_param[i];

      total_error += dist;
      n_visiables_++;
    }
  }

  top[0]->mutable_cpu_data()[0] = total_error / n_visiables_;
}

template <typename Dtype>
void NormalizeLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* predict = bottom[0]->cpu_data();
  const Dtype* ground_truth = bottom[1]->cpu_data();
  const Dtype* visiable = bottom[2]->cpu_data();
  const Dtype* normalize_param = bottom[3]->cpu_data();

  int batch_size = bottom[0]->shape(0);
  int sample_size = bottom[0]->shape(1);
  int pts = sample_size / 2;

  // only calculate predict diff
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype top_diff = top[0]->cpu_diff()[0];

  for (int i = 0; i < batch_size; ++i) {
    Dtype K = this->n_visiables_ * normalize_param[i];
    for (int j = 0; j < pts; ++j) {
      Dtype p_x = predict[i * sample_size + j * 2];
      Dtype p_y = predict[i * sample_size + j * 2 + 1];

      Dtype g_x = ground_truth[i * sample_size + j * 2];
      Dtype g_y = ground_truth[i * sample_size + j * 2 + 1];

      Dtype v = visiable[i * pts + j];
      if (v <= 0.5) {
        // no diff
        bottom_diff[i * sample_size + j * 2] = 0;
        bottom_diff[i * sample_size + j * 2 + 1] = 0;
      } else {
        Dtype dist =
            sqrtf((p_x - g_x) * (p_x - g_x) + (p_y - g_y) * (p_y - g_y));
        bottom_diff[i * sample_size + j * 2] = (p_x - g_x) / dist / K * top_diff;
        bottom_diff[i * sample_size + j * 2 + 1] = (p_y - g_y) / dist / K * top_diff;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormalizeLossLayer);
#endif

INSTANTIATE_CLASS(NormalizeLossLayer);
REGISTER_LAYER_CLASS(NormalizeLoss);

}  // namespace caffe
