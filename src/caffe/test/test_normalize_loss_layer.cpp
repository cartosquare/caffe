#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

/*
ground-truth
image_id,image_category,waistband_left,waistband_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out
0000001.jpg,trousers,430_284_0,713_303_0,560_537_1,560_626_1,361_588_1,573_622_1,-1_-1_-1
0000002.jpg,trousers,359_301_1,464_297_1,417_403_1,340_669_1,308_658_1,456_713_1,491_714_1

predict
image_id,image_category,waistband_left,waistband_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out
0000001.jpg,trousers,430_294_0,713_323_0,560_567_1,560_666_1,361_638_1,573_682_1,123_345_1
0000002.jpg,trousers,359_311_1,464_317_1,417_433_1,340_709_1,308_708_1,456_773_1,491_784_1

normalize-param:
上衣、外套、连衣裙为两个腋窝点欧式距离，裤子和半身裙为两个裤头点的欧式距离
对于第一张图，归一化参数为283.64，第二张图的归一化参数为105.0

normalize-dist:
0000001.jpg,0,0,0.105, 0.141, 0.176, 0.212,0
0000002.jpg,0.095, 0.190, 0.285, 0.381, 0.476, 0.570,0.666

NE=30.00%
*/
template <typename TypeParam>
class NormalizeLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NormalizeLossLayerTest()
      : blob_bottom_predict_(new Blob<Dtype>(2, 14, 1, 1)),
        blob_bottom_groundtruth_(new Blob<Dtype>(2, 14, 1, 1)),
        blob_bottom_visibility_(new Blob<Dtype>(2, 7, 1, 1)),
        blob_bottom_normalize_param_(new Blob<Dtype>(2, 1, 1, 1)),

        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Dtype batch_gt[2][21] = {{430, 284, 0,   713, 303, 0,   560, 537, 1,  560, 626,
                           1,   361, 588, 1,   573, 622, 1,   -1,  -1, -1},
                          {359, 301, 1, 464, 297, 1, 417, 403, 1, 340, 669, 1,
                           308, 658, 1, 456, 713, 1, 491, 714, 1}};

    Dtype batch_predict[2][21] = {
        {430, 294, 0,   713, 323, 0,   560, 567, 1,   560, 666,
         1,   361, 638, 1,   573, 682, 1,   123, 345, 1},
        {359, 311, 1,   464, 317, 1,   417, 433, 1,   340, 709,
         1,   308, 708, 1,   456, 773, 1,   491, 784, 1}};

    Dtype* predict = blob_bottom_predict_->mutable_cpu_data();
    Dtype* groundtruth = blob_bottom_groundtruth_->mutable_cpu_data();
    Dtype* visibility = blob_bottom_visibility_->mutable_cpu_data();
    Dtype* normalize_param = blob_bottom_normalize_param_->mutable_cpu_data();

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 7; ++j) {
        predict[i * 14 + j * 2] = batch_predict[i][j * 3];
        predict[i * 14 + j * 2 + 1] = batch_predict[i][j * 3 + 1];

        groundtruth[i * 14 + j * 2] = batch_gt[i][j * 3];
        groundtruth[i * 14 + j * 2 + 1] = batch_gt[i][j * 3 + 1];

        visibility[i * 7 + j] = batch_gt[i][j * 3 + 2];
      }
    }
    normalize_param[0] = 283.64;
    normalize_param[1] = 105.0;

    blob_bottom_vec_.push_back(blob_bottom_predict_);
    blob_bottom_vec_.push_back(blob_bottom_groundtruth_);
    blob_bottom_vec_.push_back(blob_bottom_visibility_);
    blob_bottom_vec_.push_back(blob_bottom_normalize_param_);

    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~NormalizeLossLayerTest() {
    delete blob_bottom_predict_;
    delete blob_bottom_groundtruth_;
    delete blob_bottom_visibility_;
    delete blob_bottom_normalize_param_;
    delete blob_top_loss_;
  }

  void TestForward() {
    typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
    NormalizeLossLayer<Dtype> layer(layer_param);

    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss = layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_NEAR(loss, 0.3, 0.01);
  }

  Blob<Dtype>* const blob_bottom_predict_;
  Blob<Dtype>* const blob_bottom_groundtruth_;
  Blob<Dtype>* const blob_bottom_visibility_;
  Blob<Dtype>* const blob_bottom_normalize_param_;

  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormalizeLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizeLossLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(NormalizeLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

}  // namespace caffe
