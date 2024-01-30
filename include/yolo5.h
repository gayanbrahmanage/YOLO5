#ifndef yolo5_H
#define yolo5_H
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <parameters.h>
#include <opencv2/opencv.hpp>

class yolo5{
private:
  // Constants.
  float INPUT_WIDTH = 640.0;
  float INPUT_HEIGHT = 640.0;
  float SCORE_THRESHOLD = 0.3;
  float NMS_THRESHOLD = 0.2;
  float CONFIDENCE_THRESHOLD = 0.3;

  // Text parameters.
  float FONT_SCALE = 0.7;
  int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
  int THICKNESS = 1;

  // Colors.
  cv::Scalar BLACK = cv::Scalar(0,0,0);
  cv::Scalar BLUE = cv::Scalar(255, 178, 50);
  cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
  cv::Scalar RED = cv::Scalar(0,0,255);

public:
  std::vector<cv::Rect> black_mask;
  std::vector<std::string> class_list;
  cv::dnn::Net net;

  yolo5();
  ~yolo5();
  void init();
  void draw_label(cv::Mat& input_image, std::string label, int left, int top);
  std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net);
  void post_process(cv::Mat &input_image,
                    std::vector<cv::Mat> &outputs,
                    const std::vector<std::string> &class_name,
                    std::vector<int>& class_ids,
                    std::vector<cv::Rect> &boxes,
                    std::vector<int>& indices);

  void detect(cv::Mat& image, std::vector<cv::Rect>& boxes, std::vector<int> &indices, std::vector<int> &class_ids,parameters* param);


};

#endif
