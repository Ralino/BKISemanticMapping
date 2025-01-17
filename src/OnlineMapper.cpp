#include "semantic_bki/OnlineMapper.hpp"

#include "semantic_bki/common/markerarray_pub.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/transforms.h>

OnlineMapper::OnlineMapper(const OnlineMapper::Params &params,
                           std::shared_ptr<tf2_ros::Buffer> tf_buffer)
    : m_params(params), m_tf_buffer(tf_buffer) {
  std::cout << "num_class: " << m_params.num_class << std::endl;
  std::cout << "method: " << (m_params.csm? "csm" : (m_params.kdtree? "kdtree" : "bki")) << std::endl;

  m_bki_map = std::make_unique<semantic_bki::SemanticBKIOctoMap>(
      m_params.resolution, m_params.block_depth, m_params.num_class,
      m_params.sf2, m_params.ell, m_params.prior, m_params.var_thresh,
      m_params.free_thresh, m_params.occupied_thresh);
  m_vis_pub =
      std::make_unique<semantic_bki::MarkerArrayPub>(m_params.resolution);

  if (!m_params.class_mapping.empty()) {
    m_bki_map->set_class_mapping(m_params.class_mapping);
  } else {
    ROS_WARN("No class mapping given, using default 1 to 1 mapping");
  }
}

OnlineMapper::~OnlineMapper() = default;

void OnlineMapper::labeledPointCloudCallback(
    const semantic_bki::PCLPointCloud &msg) {
  auto time = pcl_conversions::fromPCL(msg.header.stamp);
  geometry_msgs::TransformStamped pc_pos;
  try {
    pc_pos = m_tf_buffer->lookupTransform(m_params.map_frame,
                                          msg.header.frame_id, time);
  } catch (tf2::TransformException &ex) {
    ROS_WARN_STREAM("Failed to get transform between lidar and "
                    << m_params.map_frame << ": " << ex.what());
    return;
  }

  pcl_ros::transformPointCloud(msg, m_cloud, pc_pos.transform);
  semantic_bki::point3f origin{
      static_cast<float>(pc_pos.transform.translation.x),
      static_cast<float>(pc_pos.transform.translation.y),
      static_cast<float>(pc_pos.transform.translation.z)};
  if (m_params.csm) {
    m_bki_map->insert_pointcloud_csm(m_cloud, origin, -1.f,
                                     m_params.free_resolution,
                                     m_params.max_range);
  } else {
    if (m_params.kdtree) {
      m_bki_map->insert_pointcloud_kdtree(m_cloud, origin, -1.f,
                                   m_params.free_resolution, m_params.max_range);
    } else {
      m_bki_map->insert_pointcloud(m_cloud, origin, -1.f,
                                   m_params.free_resolution, m_params.max_range);
    }
  }
}

void OnlineMapper::visualize(const ros::Publisher &publisher) {
  m_vis_pub->clear_map(m_params.resolution);
  for (auto it = m_bki_map->begin_leaf(); it != m_bki_map->end_leaf(); ++it) {
    if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
      semantic_bki::point3f p = it.get_loc();
      m_vis_pub->insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(),
                                          it.get_node().get_semantics(),
                                          semantic_bki::ColorMap::RELLIS);
    }
  }
  m_vis_pub->publish(publisher);
}
