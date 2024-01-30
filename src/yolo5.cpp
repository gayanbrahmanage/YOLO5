#include <yolo5.h>

yolo5::yolo5(){

}
yolo5::~yolo5(){

}

void yolo5::detect(cv::Mat& image,  std::vector<cv::Rect>& boxes, std::vector<int> &indices, std::vector<int> &class_ids,parameters* param){

  std::vector<cv::Mat> detections;
  detections = pre_process(image, net);
  post_process(image, detections, class_list,class_ids,boxes,indices);

}

void yolo5::init(){

  std::ifstream ifs("../YOLOv5/coco.names");
  std::string line;
  while (getline(ifs, line))
  {
      class_list.push_back(line);
  }

  net = cv::dnn::readNet("../YOLOv5/YOLOv5s.onnx");

}

void yolo5::draw_label(cv::Mat& input_image, std::string label, int left, int top){
    // Display the label at the top of the bounding box.
    int baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
    // Put the label on the black rectangle.
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

std::vector<cv::Mat> yolo5::pre_process(cv::Mat &input_image, cv::dnn::Net &net){

    //remove black mask area
    for(std::vector<cv::Rect>::iterator it=black_mask.begin(); it!=black_mask.end();++it){
      cv::Rect r=*it;
      cv::Mat mask=cv::Mat(r.height,r.width, CV_8UC3,cv::Scalar(0,0,0));
      mask.copyTo(input_image(r));
    }

    //cv::imshow("3DMonoDetector", input_image);
    //cv::waitKey( 0 );

    // Convert to blob.
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


void yolo5::post_process(cv::Mat &input_image,
                         std::vector<cv::Mat> &outputs,
                         const std::vector<std::string> &class_name,
                         std::vector<int>& class_ids,
                         std::vector<cv::Rect> &boxes,
                         std::vector<int>& indices){

    // Initialize vectors to hold respective outputs while unwrapping     detections.

    //std::cout<<outputs.size()<<std::endl;
    //std::cout<<class_list.size()<<std::endl;
    boxes.clear();
    indices.clear();


    class_ids.clear();
    std::vector<float> confidences;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
    const int dimensions = 85;
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];


                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 85;
    }
    // Perform Non-Maximum Suppression and draw predictions.

    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

/*
    for (int i = 0; i < indices.size(); i++){

        int idx = indices[i];
        cv::Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;

        // Draw bounding box.
        cv::rectangle(input_image, box, BLUE, THICKNESS);

        //std::cout<<left<<" "<<top<<" "<< width<<" "<<height<<std::endl;
        // Get the label for the class name and its confidence.
        std::string label = cv::format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    */

}
