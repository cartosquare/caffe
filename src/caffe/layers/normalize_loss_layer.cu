#include <vector>

#include "caffe/layers/normalize_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      const Dtype* predict = bottom[0]->gpu_data();
      const Dtype* ground_truth = bottom[1]->gpu_data();
      const Dtype* visiable = bottom[2]->gpu_data();
      const Dtype* normalize_param = bottom[3]->gpu_data();
    
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
void NormalizeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      const Dtype* predict = bottom[0]->gpu_data();
      const Dtype* ground_truth = bottom[1]->gpu_data();
      const Dtype* visiable = bottom[2]->gpu_data();
      const Dtype* normalize_param = bottom[3]->gpu_data();
    
      int batch_size = bottom[0]->shape(0);
      int sample_size = bottom[0]->shape(1);
      int pts = sample_size / 2;
    
      // only calculate predict diff
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype top_diff = top[0]->gpu_diff()[0];
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
            bottom_diff[i * sample_size + j * 2] = p_x / dist / K * top_diff;
            bottom_diff[i * sample_size + j * 2 + 1] = p_y / dist / K * top_diff;
          }
        }
      }
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLossLayer);

}  // namespace caffe
