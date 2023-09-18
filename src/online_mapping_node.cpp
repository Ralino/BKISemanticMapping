#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>

#include "bkioctomap.h"
#include "markerarray_pub.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

struct Params {
  float resolution = 0.1;
  int block_depth = 3;
  int num_class = 20;
  float sf2 = 10;
  float ell = 0.3;
  float prior = 0.001;
  float var_thresh = 0.09;     // unused
  float free_thresh = 0.3;     // unused
  float occupied_thresh = 0.7; // unused

  float ds_resolution = 0.1;
  float free_resolution = 100; // beam traversal for points of class empty.
                               // Value of 100 effectively disables this
  float max_range = -1;

  bool csm = false;
};

class OnlineMapper {
public:
  OnlineMapper() : m_nh(""), m_ph("~"), m_tf_listener(m_tf_buffer) {
    m_labeled_pc_sub = m_nh.subscribe(
        "ground_truth/xyzl", 1, &OnlineMapper::labeledPointCloudCallback, this);

    m_ph.getParam("resolution", m_params.resolution);
    m_ph.getParam("block_depth", m_params.block_depth);
    m_ph.getParam("num_class", m_params.num_class);
    m_ph.getParam("sf2", m_params.sf2);
    m_ph.getParam("ell", m_params.ell);
    m_ph.getParam("prior", m_params.prior);
    m_ph.getParam("var_thresh", m_params.var_thresh);
    m_ph.getParam("free_thresh", m_params.free_thresh);
    m_ph.getParam("occupied_thresh", m_params.occupied_thresh);

    m_ph.getParam("ds_resolution", m_params.ds_resolution);
    m_ph.getParam("free_resolution", m_params.free_resolution);
    m_ph.getParam("max_range", m_params.max_range);
    m_ph.getParam("csm", m_params.csm);

    std::cout << "num_class: " << m_params.num_class << std::endl;
    std::cout << "sizeof block: " << sizeof(semantic_bki::Block) << std::endl;

    m_bki_map = std::make_unique<semantic_bki::SemanticBKIOctoMap>(
        m_params.resolution, m_params.block_depth, m_params.num_class,
        m_params.sf2, m_params.ell, m_params.prior, m_params.var_thresh,
        m_params.free_thresh, m_params.occupied_thresh);
    m_vis_pub = std::make_unique<semantic_bki::MarkerArrayPub>(
        m_ph, "occupied_cells_vis_array", m_params.resolution);
  }

  void labeledPointCloudCallback(const semantic_bki::PCLPointCloud &msg) {
    // µs to s and ns
    auto time = ros::Time(msg.header.stamp / 1000'000,
                          (msg.header.stamp % 1000'000) * 1000);
    geometry_msgs::TransformStamped pc_pos;
    try {
      pc_pos = m_tf_buffer.lookupTransform("map", msg.header.frame_id, time);
    } catch (tf2::TransformException &ex) {
      ROS_WARN_STREAM(
          "Failed to get transform between lidar and map: " << ex.what());
      return;
    }

    pcl_ros::transformPointCloud(msg, m_cloud, pc_pos.transform);
    semantic_bki::point3f origin{
        static_cast<float>(pc_pos.transform.translation.x),
        static_cast<float>(pc_pos.transform.translation.y),
        static_cast<float>(pc_pos.transform.translation.z)};
    if (m_params.csm) {
      m_bki_map->insert_pointcloud_csm(m_cloud, origin, m_params.ds_resolution,
                                       m_params.free_resolution,
                                       m_params.max_range);
    } else {
      m_bki_map->insert_pointcloud(m_cloud, origin, m_params.ds_resolution,
                                   m_params.free_resolution,
                                   m_params.max_range);
    }

    visualize();
  }

  void visualize() {
    m_vis_pub->clear_map(m_params.resolution);
    for (auto it = m_bki_map->begin_leaf(); it != m_bki_map->end_leaf(); ++it) {
      if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
        semantic_bki::point3f p = it.get_loc();
        m_vis_pub->insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(),
                                            it.get_node().get_semantics(),
                                            semantic_bki::ColorMap::RELLIS);
      }
    }
    m_vis_pub->publish();
  }

private:
  Params m_params;

  ros::NodeHandle m_nh;
  ros::NodeHandle m_ph;
  ros::Subscriber m_labeled_pc_sub;
  tf2_ros::Buffer m_tf_buffer;
  tf2_ros::TransformListener m_tf_listener;

  std::unique_ptr<semantic_bki::SemanticBKIOctoMap> m_bki_map;
  std::unique_ptr<semantic_bki::MarkerArrayPub> m_vis_pub;

  semantic_bki::PCLPointCloud m_cloud;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "online_mapping_node");
  OnlineMapper mapper;
  ros::spin();
}
