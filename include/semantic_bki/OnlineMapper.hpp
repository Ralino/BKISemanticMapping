#pragma once

#include "semantic_bki/mapping/bkioctomap.h"

#include <pcl_ros/point_cloud.h>
#include <tf2_ros/buffer.h>

namespace semantic_bki {
class MarkerArrayPub;
}

class OnlineMapper {
public:
  struct Params {
    float resolution = 0.3;
    int block_depth = 1;
    int num_class = 35;
    float sf2 = 10;
    float ell = 0.3;
    float prior = 0.001; // no effect, because positive definite and max
    float var_thresh = 0.09;     // unused
    float free_thresh = 0.3;     // unused
    float occupied_thresh = 0.7; // unused

    float free_resolution = 2.; // beam traversal for points of class empty.
    float max_range = 10.;

    std::vector<uint32_t> class_mapping;
    bool csm = false;
    bool kdtree = false;
  };

  OnlineMapper(const Params &params,
               std::shared_ptr<tf2_ros::Buffer> tf_buffer);

  ~OnlineMapper();

  void labeledPointCloudCallback(const semantic_bki::PCLPointCloud &msg);

  void visualize(const ros::Publisher &publisher);

private:
  Params m_params;

  std::shared_ptr<tf2_ros::Buffer> m_tf_buffer;

  std::unique_ptr<semantic_bki::SemanticBKIOctoMap> m_bki_map;
  std::unique_ptr<semantic_bki::MarkerArrayPub> m_vis_pub;

  semantic_bki::PCLPointCloud m_cloud;
};
