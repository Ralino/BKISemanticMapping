#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

#include "semantic_bki/OnlineMapper.hpp"

int main(int argc, char **argv) {
  ros::init(argc, argv, "online_mapping_node");
  ros::NodeHandle nh;
  ros::NodeHandle ph("~");

  ros::Subscriber labeled_pc_sub;

  OnlineMapper::Params params;
  ph.getParam("resolution", params.resolution);
  ph.getParam("block_depth", params.block_depth);
  ph.getParam("num_class", params.num_class);
  ph.getParam("sf2", params.sf2);
  ph.getParam("ell", params.ell);
  ph.getParam("prior", params.prior);
  ph.getParam("var_thresh", params.var_thresh);
  ph.getParam("free_thresh", params.free_thresh);
  ph.getParam("occupied_thresh", params.occupied_thresh);

  ph.getParam("free_resolution", params.free_resolution);
  ph.getParam("max_range", params.max_range);
  ph.getParam("csm", params.csm);
  ph.getParam("kdtree", params.kdtree);

  std::vector<int> signed_class_mapping;
  ph.getParam("class_mapping", signed_class_mapping);
  params.class_mapping.clear();
  for (int label : signed_class_mapping) {
    if (label < 0) {
      // disable negative ones and log a warning if they appear
      params.class_mapping.push_back(params.num_class);
    } else {
      params.class_mapping.push_back(static_cast<uint32_t>(label));
    }
  }

  auto tf_buffer = std::make_shared<tf2_ros::Buffer>();
  tf2_ros::TransformListener tf_listener(*tf_buffer);
  ros::Publisher vis_pub = ph.advertise<visualization_msgs::MarkerArray>(
      "occupied_cells_vis_array", 1, true);

  OnlineMapper mapper(params, tf_buffer);
  labeled_pc_sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZL>>(
      "ground_truth/xyzl", 1,
      [&](const pcl::PointCloud<pcl::PointXYZL>::ConstPtr &msg) {
        mapper.labeledPointCloudCallback(*msg);
        mapper.visualize(vis_pub);
      });
  ros::spin();
}
