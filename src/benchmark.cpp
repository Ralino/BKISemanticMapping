#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf2_ros/buffer.h>

#include "OnlineMapper.hpp"
#include "tf2_msgs/TFMessage.h"

static constexpr int NUM_FRAMES = 200;

class MultiViewIterator {
public:
  MultiViewIterator(const std::vector<std::unique_ptr<rosbag::Bag>> &bags,
                    const std::string &topic_name) {
    for (const auto &bag : bags) {
      auto view_ptr =
          std::make_unique<rosbag::View>(*bag, rosbag::TopicQuery(topic_name));
      m_views.emplace_back(std::move(view_ptr));
    }
    std::transform(m_views.begin(), m_views.end(),
                   std::back_inserter(m_iterators),
                   [](auto &view) { return view->begin(); });
    setIndexToEarliest();
  }

  bool atEnd() {
    return m_iterators[m_current_index] == m_views[m_current_index]->end();
  }

  MultiViewIterator &operator++() {
    m_total_read_msgs++;
    m_iterators[m_current_index]++;
    setIndexToEarliest();
    return *this;
  }

  rosbag::MessageInstance *operator->() {
    return m_iterators[m_current_index].operator->();
  }

  size_t getNrOfTotalReadMsgs() const { return m_total_read_msgs; }

private:
  void setIndexToEarliest() {
    for (size_t i = 0; i < m_views.size(); ++i) {
      if (m_iterators[i] == m_views[i]->end()) {
        continue;
      }
      if (m_iterators[m_current_index] == m_views[m_current_index]->end() ||
          m_iterators[i]->getTime() < m_iterators[m_current_index]->getTime()) {
        m_current_index = i;
      }
    }
  }
  std::vector<std::unique_ptr<rosbag::View>> m_views;
  std::vector<rosbag::View::iterator> m_iterators;
  size_t m_current_index = 0;
  size_t m_total_read_msgs = 0;
};

void printHelp(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " <bagfile.bag>..." << std::endl;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "bki_benchmark");
  if (argc < 2) {
    printHelp(argv[0]);
    return 1;
  }

  std::vector<std::unique_ptr<rosbag::Bag>> bags;

  for (int i = 1; i < argc; ++i) {
    std::cout << "opening bag '" << argv[i] << "'... " << std::endl;
    auto bag_ptr = std::make_unique<rosbag::Bag>();
    bags.emplace_back(std::move(bag_ptr));
    try {
      bags.back()->open(argv[i], rosbag::bagmode::Read);
    } catch (rosbag::BagException e) {
      std::cout << "Failed to open '" << argv[i]
                << "' as a bag file: " << e.what() << std::endl;
      printHelp(argv[0]);
      return 1;
    }
  }

  auto tf_buffer = std::make_shared<tf2_ros::Buffer>();

  for (MultiViewIterator tf_static_iterator(bags, "/tf_static");
       !tf_static_iterator.atEnd(); ++tf_static_iterator) {
    auto tf_msg_ptr = tf_static_iterator->instantiate<tf2_msgs::TFMessage>();
    if (tf_msg_ptr) {
      for (const auto &tf : tf_msg_ptr->transforms) {
        tf_buffer->setTransform(tf, "tf_static", /*is_static=*/true);
      }
    }
  }

  MultiViewIterator pc_msg_iterator(bags, "/ground_truth/xyzl");
  MultiViewIterator tf_msg_iterator(bags, "/tf");

  OnlineMapper::Params params;
  OnlineMapper mapper(params, tf_buffer);

  int frame_number;
  auto start_time = std::chrono::steady_clock::now();
  for (frame_number = 0; frame_number < NUM_FRAMES; ++frame_number) {
    if (pc_msg_iterator.atEnd()) {
      frame_number++;
      break;
    }
    auto current_frame_time = pc_msg_iterator->getTime();

    while (!tf_msg_iterator.atEnd() &&
           tf_msg_iterator->getTime() <= current_frame_time) {
      auto tf_msg_ptr = tf_msg_iterator->instantiate<tf2_msgs::TFMessage>();
      if (tf_msg_ptr) {
        for (const auto &tf : tf_msg_ptr->transforms) {
          tf_buffer->setTransform(tf, "tf");
        }
      }
      ++tf_msg_iterator;
    }
    auto pc_msg_ptr =
        pc_msg_iterator->instantiate<pcl::PointCloud<pcl::PointXYZL>>();
    if (!pc_msg_ptr) {
      std::cout << "A message from topic 'ground_truth/xyzl' could not be "
                   "converted to SemanticPointCloud"
                << std::endl;
      return 1;
    }
    mapper.labeledPointCloudCallback(*pc_msg_ptr);

    ++pc_msg_iterator;
  }
  auto end_time = std::chrono::steady_clock::now();
  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  std::cout << "Total runtime (" << frame_number
            << " frames): " << delta.count() << "ms" << std::endl;
  std::cout << "Per frame: " << delta.count() / frame_number << "ms"
            << std::endl;
}
