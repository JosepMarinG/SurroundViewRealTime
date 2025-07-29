// surround_view_ROS.cpp
// Modificación del programa original para publicar en ROS2 con imagen comprimida

#include "common.h"
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>


#define AWB_LUN_BANLANCE_ENALE 1

class SurroundViewNode : public rclcpp::Node {
public:
    SurroundViewNode(const std::string &data_path)
    : Node("surround_view_node"), data_path_(data_path) {
        image_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>("/surround_view/image/compressed", 10);

        // Inicializar parámetros y cámaras
        init();
        loop();
    }

private:
    std::string data_path_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr image_pub_;
    cv::VideoCapture cams[4];
    CameraPrms prms[4];
    cv::Mat car_img, out_put_img;
    cv::Mat origin_dir_img[4], undist_dir_img[4];
    cv::Mat merge_weights_img[4];
    float *w_ptr[4];

    void init() {
        car_img = cv::imread(data_path_ + "/images/car.png");
        cv::resize(car_img, car_img, cv::Size(xr - xl, yb - yt));
        out_put_img = cv::Mat(cv::Size(total_w, total_h), CV_8UC3, cv::Scalar(0, 0, 0));

        cv::Mat weights = cv::imread(data_path_ + "/yaml/weights.png", -1);
        for (int i = 0; i < 4; ++i) {
            merge_weights_img[i] = cv::Mat(weights.size(), CV_32FC1, cv::Scalar(0));
            w_ptr[i] = (float *)merge_weights_img[i].data;
        }

        int pixel_index = 0;
        for (int h = 0; h < weights.rows; ++h) {
            uchar* uc_pixel = weights.data + h * weights.step;
            for (int w = 0; w < weights.cols; ++w) {
                for (int i = 0; i < 4; ++i)
                    w_ptr[i][pixel_index] = uc_pixel[i] / 255.0f;
                uc_pixel += 4;
                ++pixel_index;
            }
        }

        for (int i = 0; i < 4; ++i) {
            prms[i].name = camera_names[i];
            if (!read_prms(data_path_ + "/yaml/" + prms[i].name + ".yaml", prms[i])) exit(-1);
        }

        int cam_ids[4] = {2, 4, 6, 0};
        for (int i = 0; i < 4; ++i) {
            cams[i].open(cam_ids[i]);
            if (!cams[i].isOpened()) exit(-1);
            cams[i].set(cv::CAP_PROP_FRAME_WIDTH, 1920);
            cams[i].set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        }
    }

    void loop() {
        while (rclcpp::ok()) {
            if (!capture_and_process()) continue;
            publish_image();
            if (cv::waitKey(1) == 'q') break;
        }
    }

    bool capture_and_process() {
        std::vector<cv::Mat*> srcs;
        for (int i = 0; i < 4; ++i) {
            cams[i] >> origin_dir_img[i];
            if (origin_dir_img[i].empty()) return false;
            srcs.push_back(&origin_dir_img[i]);
        }
#if AWB_LUN_BANLANCE_ENALE
        awb_and_lum_banlance(srcs);
#endif

        for (int i = 0; i < 4; ++i) {
            auto &src = origin_dir_img[i];
            undist_by_remap(src, src, prms[i]);
            cv::warpPerspective(src, src, prms[i].project_matrix, project_shapes[prms[i].name]);
            if (camera_flip_mir[i] == "r+")
                cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
            else if (camera_flip_mir[i] == "r-")
                cv::rotate(src, src, cv::ROTATE_90_COUNTERCLOCKWISE);
            else if (camera_flip_mir[i] == "m")
                cv::rotate(src, src, cv::ROTATE_180);
            undist_dir_img[i] = src.clone();
        }

        car_img.copyTo(out_put_img(cv::Rect(xl, yt, car_img.cols, car_img.rows)));
        for (int i = 0; i < 4; ++i) {
            std::string name = camera_names[i];
            if (name == "front")
                undist_dir_img[i](cv::Rect(xl, 0, xr - xl, yt)).copyTo(out_put_img(cv::Rect(xl, 0, xr - xl, yt)));
            else if (name == "left")
                undist_dir_img[i](cv::Rect(0, yt, xl, yb - yt)).copyTo(out_put_img(cv::Rect(0, yt, xl, yb - yt)));
            else if (name == "right")
                undist_dir_img[i](cv::Rect(0, yt, xl, yb - yt)).copyTo(out_put_img(cv::Rect(xr, yt, total_w - xr, yb - yt)));
            else if (name == "back")
                undist_dir_img[i](cv::Rect(xl, 0, xr - xl, yt)).copyTo(out_put_img(cv::Rect(xl, yb, xr - xl, yt)));
        }

        merge_image(undist_dir_img[0](cv::Rect(0, 0, xl, yt)), undist_dir_img[1](cv::Rect(0, 0, xl, yt)), merge_weights_img[2], out_put_img(cv::Rect(0, 0, xl, yt)));
        merge_image(undist_dir_img[0](cv::Rect(xr, 0, xl, yt)), undist_dir_img[3](cv::Rect(0, 0, xl, yt)), merge_weights_img[1], out_put_img(cv::Rect(xr, 0, xl, yt)));
        merge_image(undist_dir_img[2](cv::Rect(0, 0, xl, yt)), undist_dir_img[1](cv::Rect(0, yb, xl, yt)), merge_weights_img[0], out_put_img(cv::Rect(0, yb, xl, yt)));
        merge_image(undist_dir_img[2](cv::Rect(xr, 0, xl, yt)), undist_dir_img[3](cv::Rect(0, yb, xl, yt)), merge_weights_img[3], out_put_img(cv::Rect(xr, yb, xl, yt)));

        return true;
    }

    void publish_image() {
        cv::Mat resized, compressed;
        cv::resize(out_put_img, resized, cv::Size(out_put_img.cols / 2, out_put_img.rows / 2));
        std::vector<uchar> buff;
        cv::imencode(".jpg", resized, buff);

        auto msg = sensor_msgs::msg::CompressedImage();
        msg.header.stamp = this->get_clock()->now();
        msg.format = "jpeg";
        msg.data = std::move(buff);
        image_pub_->publish(msg);
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "usage:\n\t" << argv[0] << " path\n";
        return -1;
    }
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SurroundViewNode>(argv[1]);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
