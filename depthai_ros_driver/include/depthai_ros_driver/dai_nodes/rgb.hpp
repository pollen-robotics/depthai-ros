#pragma once

#include "depthai/depthai.hpp"
#include "depthai_ros_driver/dai_nodes/base_node.hpp"
#include "depthai_ros_driver/param_handlers/rgb_param_handler.hpp"
#include "image_transport/camera_publisher.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace depthai_ros_driver {
namespace dai_nodes {
namespace link_types {
enum class RGBLinkType { color, preview };
};
class RGB : public BaseNode {
   public:
    explicit RGB(const std::string& dai_node_name, rclcpp::Node* node, std::shared_ptr<dai::Pipeline> pipeline);
    virtual ~RGB() = default;
    void updateParams(const std::vector<rclcpp::Parameter>& params) override;
    void setupQueues(std::shared_ptr<dai::Device> device) override;
    void link(const dai::Node::Input& in, int link_type = 0) override;
    dai::Node::Input get_input(int link_type = 0) override;
    void set_names() override;
    void set_xin_xout(std::shared_ptr<dai::Pipeline> pipeline) override;

   private:
    void color_q_cb(const std::string& name, const std::shared_ptr<dai::ADatatype>& data);
    image_transport::CameraPublisher rgb_pub_, preview_pub_;
    sensor_msgs::msg::CameraInfo rgb_info_, preview_info_;
    std::shared_ptr<dai::node::ColorCamera> color_cam_node_;
    std::unique_ptr<param_handlers::RGBParamHandler> param_handler_;
    std::shared_ptr<dai::DataOutputQueue> color_q_, preview_q_;
    std::shared_ptr<dai::DataInputQueue> control_q_;
    std::shared_ptr<dai::node::XLinkOut> xout_color_, xout_preview_;
    std::shared_ptr<dai::node::XLinkIn> xin_control_;
    std::string color_q_name_, preview_q_name_, control_q_name_;
};
class RGBFactory : public BaseNodeFactory {
   public:
    std::unique_ptr<BaseNode> create(const std::string& dai_node_name, rclcpp::Node* node, std::shared_ptr<dai::Pipeline> pipeline) {
        return std::make_unique<RGB>(dai_node_name, node, pipeline);
    };
};
}  // namespace dai_nodes
}  // namespace depthai_ros_driver