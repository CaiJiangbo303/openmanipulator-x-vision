#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

using Clock = std::chrono::steady_clock;
using Image = sensor_msgs::msg::Image;

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("d435_cpp_probe");

    std::vector<double> rgb, depth;
    auto start = Clock::now();

    auto record = [&](std::vector<double>& data) {
        double t = std::chrono::duration<double>(
            Clock::now() - start).count();
        if (t >= 2.0 && t < 17.0)
            data.push_back(t);
    };

    auto qos = rclcpp::SensorDataQoS();

    auto s1 = node->create_subscription<Image>(
        "/camera/camera/color/image_raw", qos,
        [&](Image::ConstSharedPtr) { record(rgb); });

    auto s2 = node->create_subscription<Image>(
        "/camera/camera/aligned_depth_to_color/image_raw", qos,
        [&](Image::ConstSharedPtr) { record(depth); });

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);

    std::cout << "Warmup 2s, measuring 15s" << std::endl;

    while (rclcpp::ok() &&
        std::chrono::duration<double>(
            Clock::now()-start).count() < 17.0) {
        executor.spin_once(std::chrono::milliseconds(20));
    }

    auto report = [](const char* name,
                     const std::vector<double>& v) {
        std::cout << "\n===== " << name << " =====\n";
        std::cout << "Frames: " << v.size() << "\n";
        if (v.size() < 2) return;

        double max_gap = 0;
        int gaps = 0;

        for (size_t i=1; i<v.size(); ++i) {
            double dt = (v[i]-v[i-1])*1000;
            max_gap = std::max(max_gap, dt);
            if (dt > 100) ++gaps;
        }

        std::cout << "Receive Hz: "
                  << (v.size()-1)/(v.back()-v.front()) << "\n";
        std::cout << "Max gap ms: " << max_gap << "\n";
        std::cout << "Gaps over 100ms: " << gaps << "\n";
    };

    report("CPP COLOR", rgb);
    report("CPP ALIGNED DEPTH", depth);
    rclcpp::shutdown();
}
