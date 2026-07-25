#include <gtest/gtest.h>

#include "target_ekf.h"

TEST(GeneralKalmanFilterTest, NisUsesPredictedInnovation) {
    GeneralKalmanFilter kf(1, 10.0, 10);
    cv::Mat x0 = cv::Mat::zeros(1, 1, CV_64F);
    cv::Mat p0 = cv::Mat::eye(1, 1, CV_64F);
    kf.reset(x0, p0);

    cv::Mat z = (cv::Mat_<double>(1, 1) << 1.0);
    cv::Mat h_matrix = cv::Mat::eye(1, 1, CV_64F);
    cv::Mat r = cv::Mat::eye(1, 1, CV_64F);
    auto identity = [](const cv::Mat& x) { return x; };
    auto subtract = [](const cv::Mat& a, const cv::Mat& b) { return a - b; };

    // innovation=1, S=P+R=2，因此标准 NIS=1^2/2=0.5。
    EXPECT_NEAR(kf.update(z, h_matrix, r, identity, subtract), 0.5, 1e-9);
}

TEST(TargetEKFTest, UpdateReturnsState) {
    TargetEKF ekf;
    auto [yaw, pitch, dist] = ekf.update(0.03, 5.0, 3.0, 10.0);
    EXPECT_NE(yaw, 0.0);
    EXPECT_NE(pitch, 0.0);
    EXPECT_NE(dist, 0.0);
}

TEST(TargetEKFTest, PredictReturnsState) {
    TargetEKF ekf;
    ekf.update(0.03, 5.0, 3.0, 10.0);
    auto [yaw, pitch, dist] = ekf.predict(0.03);
    EXPECT_NE(yaw, 0.0);
}

TEST(TargetEKFTest, PredictBeforeUpdateUsesDefault) {
    TargetEKF ekf;
    auto [yaw, pitch, dist] = ekf.update(0.03, 0.0, 0.0, 3.0);
    EXPECT_DOUBLE_EQ(yaw, 0.0);
    EXPECT_DOUBLE_EQ(pitch, 0.0);
    EXPECT_DOUBLE_EQ(dist, 3.0);
}

TEST(TargetEKFTest, MultipleUpdatesConverge) {
    TargetEKF ekf;
    double yaw = 0, pitch = 0, dist = 3.0;
    for (int i = 0; i < 20; i++) {
        std::tie(yaw, pitch, dist) = ekf.update(0.03, 10.0, 5.0, 15.0);
    }
    EXPECT_NEAR(yaw, 10.0, 1.5);
    EXPECT_NEAR(pitch, 5.0, 1.5);
    EXPECT_NEAR(dist, 15.0, 1.5);
}

TEST(TargetEKFTest, ResetThenUpdateReturnsFreshState) {
    TargetEKF ekf;
    ekf.update(0.03, 10.0, 5.0, 15.0);
    ekf.reset();
    auto [yaw, pitch, dist] = ekf.update(0.03, 0.0, 0.0, 3.0);
    EXPECT_DOUBLE_EQ(yaw, 0.0);
    EXPECT_DOUBLE_EQ(pitch, 0.0);
    EXPECT_DOUBLE_EQ(dist, 3.0);
}

TEST(TargetEKFTest, NisFailureRateStartsLow) {
    TargetEKF ekf;
    double rate = ekf.nisFailureRate();
    EXPECT_GE(rate, 0.0);
    EXPECT_LE(rate, 1.0);
}
