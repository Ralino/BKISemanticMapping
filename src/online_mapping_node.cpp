#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>

#include "OnlineMapper.hpp"

int main(int argc, char **argv) {
  ros::init(argc, argv, "online_mapping_node");
  ros::NodeHandle nh;
  ros::NodeHandle ph("~");

  ros::Subscriber labeled_pc_sub;

  OnlineMapper::Params params;
  ph.getParam("resolution", params.resolution);
  ph.getParam("block_depth", params.block_depth);
  ph.getParam("nuclass", params.num_class);
  ph.getParam("sf2", params.sf2);
  ph.getParam("ell", params.ell);
  ph.getParam("prior", params.prior);
  ph.getParam("var_thresh", params.var_thresh);
  ph.getParam("free_thresh", params.free_thresh);
  ph.getParam("occupied_thresh", params.occupied_thresh);

  ph.getParam("ds_resolution", params.ds_resolution);
  ph.getParam("free_resolution", params.free_resolution);
  ph.getParam("max_range", params.max_range);
  ph.getParam("csm", params.csm);

  auto tf_buffer = std::make_shared<tf2_ros::Buffer>();
  tf2_ros::TransformListener tf_listener(*tf_buffer);

  OnlineMapper mapper(params, tf_buffer);
  labeled_pc_sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZL>>(
      "ground_truth/xyzl", 1,
      [&](const pcl::PointCloud<pcl::PointXYZL>::ConstPtr &msg) {
        mapper.labeledPointCloudCallback(*msg);
        mapper.visualize();
      });
  ros::spin();
}
