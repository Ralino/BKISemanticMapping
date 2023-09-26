#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include <unistd.h>
#include <stdio.h>
#include <yaml-cpp/yaml.h>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2_ros/buffer.h>

#include "semantic_bki/OnlineMapper.hpp"

// #define VISUALIZE

#ifdef VISUALIZE
#include <visualization_msgs/MarkerArray.h>
#endif

static constexpr int SKIP_FRAMES = 200;
static constexpr int NUM_FRAMES = 200;

namespace fs = std::filesystem;

class MultiBagIterator {
public:
  MultiBagIterator(const std::vector<std::unique_ptr<rosbag::Bag>> &bags,
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

  MultiBagIterator &operator++() {
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

void updateParamsFromYaml(YAML::Node &&config, OnlineMapper::Params &params) {
    params.resolution = config["resolution"] ? config["resolution"].as<float>()
                                             : params.resolution;
    params.block_depth = config["block_depth"] ? config["block_depth"].as<int>()
                                               : params.block_depth;
    params.num_class =
        config["num_class"] ? config["num_class"].as<int>() : params.num_class;
    params.sf2 = config["sf2"] ? config["sf2"].as<float>() : params.sf2;
    params.ell = config["ell"] ? config["ell"].as<float>() : params.ell;
    params.prior = config["prior"] ? config["prior"].as<float>() : params.prior;
    params.var_thresh = config["var_thresh"] ? config["var_thresh"].as<float>()
                                             : params.var_thresh;
    params.free_thresh = config["free_thresh"]
                             ? config["free_thresh"].as<float>()
                             : params.free_thresh;
    params.occupied_thresh = config["occupied_thresh"]
                                 ? config["occupied_thresh"].as<float>()
                                 : params.occupied_thresh;

    params.free_resolution = config["free_resolution"]
                                 ? config["free_resolution"].as<float>()
                                 : params.free_resolution;
    params.max_range = config["max_range"] ? config["max_range"].as<float>()
                                           : params.max_range;
    params.csm = config["csm"] ? config["csm"].as<bool>() : params.csm;
    params.kdtree =
        config["kdtree"] ? config["kdtree"].as<bool>() : params.kdtree;

    if (config["class_mapping"]) {
      params.class_mapping.clear();
      for (const auto &entry : config["class_mapping"]) {
        int label = entry.as<int>();
        if (label < 0) {
          // disable negative ones and log a warning if they appear
          params.class_mapping.push_back(params.num_class);
        } else {
          params.class_mapping.push_back(static_cast<uint32_t>(label));
        }
      }
    }
}

OnlineMapper::Params parseYamlFiles(const std::vector<std::string> &filenames) {
  OnlineMapper::Params params;
  for (const auto &filename : filenames) {
    updateParamsFromYaml(YAML::LoadFile(filename), params);
  }

  return params;
}

void printHelp(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " <bagfile.bag>... [<config.yaml>...]"
            << std::endl;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "bki_benchmark");
  if (argc < 2) {
    printHelp(argv[0]);
    return 1;
  }

#ifdef VISUALIZE
  ros::NodeHandle ph("~");
  ros::Publisher vis_pub = ph.advertise<visualization_msgs::MarkerArray>(
      "occupied_cells_vis_array", 1, true);

#endif

  std::vector<std::unique_ptr<rosbag::Bag>> bags;
  std::vector<std::string> config_files;

  for (int i = 1; i < argc; ++i) {
    fs::path filename = argv[i];
    if (filename.extension().string() == ".bag") {
      std::cout << "opening bag '" << argv[i] << "'... " << std::endl;
      auto bag_ptr = std::make_unique<rosbag::Bag>();
      bags.emplace_back(std::move(bag_ptr));
      try {
        bags.back()->open(argv[i], rosbag::bagmode::Read);
      } catch (rosbag::BagException e) {
        std::cout << "Failed to open '" << argv[i]
                  << "' as a bag file: " << e.what() << std::endl;
        return 1;
      }
    } else if (filename.extension().string() == ".yaml" ||
               filename.extension().string() == ".yml") {
      config_files.push_back(filename.string());
    } else {
      std::cout << "Invalid argument: " << argv[i] << std::endl;
      printHelp(argv[0]);
      return 1;
    }
  }

  auto tf_buffer = std::make_shared<tf2_ros::Buffer>();

  for (MultiBagIterator tf_static_iterator(bags, "/tf_static");
       !tf_static_iterator.atEnd(); ++tf_static_iterator) {
    auto tf_msg_ptr = tf_static_iterator->instantiate<tf2_msgs::TFMessage>();
    if (tf_msg_ptr) {
      for (const auto &tf : tf_msg_ptr->transforms) {
        tf_buffer->setTransform(tf, "tf_static", /*is_static=*/true);
      }
    }
  }

  MultiBagIterator pc_msg_iterator(bags, "/ground_truth/xyzl");
  MultiBagIterator tf_msg_iterator(bags, "/tf");

  auto params = parseYamlFiles(config_files);
  if (!isatty(fileno(stdin))) {
    updateParamsFromYaml(YAML::Load(std::cin), params);
  }
  OnlineMapper mapper(params, tf_buffer);

  int frame_number;
  std::chrono::steady_clock::time_point start_time;
  for (frame_number = 0; frame_number < SKIP_FRAMES + NUM_FRAMES;
       ++frame_number) {
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
    if (frame_number == SKIP_FRAMES) {
      start_time = std::chrono::steady_clock::now();
    }
    if (frame_number >= SKIP_FRAMES) {
      auto pc_msg_ptr =
          pc_msg_iterator->instantiate<pcl::PointCloud<pcl::PointXYZL>>();
      if (!pc_msg_ptr) {
        std::cout << "A message from topic 'ground_truth/xyzl' could not be "
                     "converted to SemanticPointCloud"
                  << std::endl;
        return 1;
      }
      mapper.labeledPointCloudCallback(*pc_msg_ptr);
#ifdef VISUALIZE
      mapper.visualize(vis_pub);
#endif
    }

    ++pc_msg_iterator;
  }
  auto end_time = std::chrono::steady_clock::now();
  auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  std::cout << "Total runtime (" << frame_number - SKIP_FRAMES
            << " frames): " << delta.count() << "ms" << std::endl;
  std::cout << "Per frame: " << delta.count() / (frame_number - SKIP_FRAMES)
            << "ms" << std::endl;
}
