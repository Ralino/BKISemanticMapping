#include <random>
#include <algorithm>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/console.h>

#include "semantic_bki/mapping/bkioctomap.h"
#include "semantic_bki/mapping/bki.h"

using std::vector;

// #define DEBUG true;

#ifdef DEBUG

#include <iostream>

#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace semantic_bki {

    SemanticBKIOctoMap::SemanticBKIOctoMap() : SemanticBKIOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        3,  // num_class
                                        1.0, // sf2
                                        1.0, // ell
                                        1.0f, // prior
                                        1.0f, // var_thresh
                                        0.3f, // free_thresh
                                        0.7f // occupied_thresh
                                    ) { }

    SemanticBKIOctoMap::SemanticBKIOctoMap(float resolution,
                        unsigned short block_depth,
                        int num_class,
                        float sf2,
                        float ell,
                        float prior,
                        float var_thresh,
                        float free_thresh,
                        float occupied_thresh)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);
        
        // Bug fixed
        Block::cell_num = static_cast<unsigned short>(round(Block::size / Block::resolution));
        std::cout << "block::resolution: " << Block::resolution << std::endl;
        std::cout << "block::size: " << Block::size << std::endl;
        std::cout << "block::cell_num: " << Block::cell_num << std::endl;
        
        SemanticOcTree::max_depth = block_depth;

        SemanticOcTreeNode::num_class = num_class;
        SemanticOcTreeNode::sf2 = sf2;
        SemanticOcTreeNode::ell = ell;
        SemanticOcTreeNode::prior = prior;
        SemanticOcTreeNode::var_thresh = var_thresh;
        SemanticOcTreeNode::free_thresh = free_thresh;
        SemanticOcTreeNode::occupied_thresh = occupied_thresh;

        m_class_mapping.reserve(num_class);
        for (int i = 0; i < num_class; ++i) {
            m_class_mapping.push_back(i);
        }
    }

    SemanticBKIOctoMap::~SemanticBKIOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void SemanticBKIOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void SemanticBKIOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        SemanticOcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void SemanticBKIOctoMap::set_class_mapping(
        const std::vector<uint32_t> &class_mapping) {
        if (class_mapping.empty() || class_mapping[0] != 0) {
            ROS_WARN("Empty class '0' cannot be remapped to another class.");
        } else {
            m_class_mapping = class_mapping;
        }
    }

    void SemanticBKIOctoMap::insert_pointcloud_csm(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);
#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif
        // If pointcloud after max_range filtering is empty
        //  no need to do anything
        if (xy.size() == 0) {
            return;
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        std::unordered_map<BlockHashKey, SemanticBKI3f *> bgk_arr;
        for (auto point_it = xy.begin(); point_it != xy.end(); ++point_it) {
            point3f pos(point_it->x, point_it->y, point_it->z);
            auto label = point_it->label;
            BlockHashKey key = block_to_hash_key(pos);

            SemanticBKI3f *bgk;
            auto bgk_it = bgk_arr.find(key);
            if (bgk_it != bgk_arr.end()) {
                bgk = bgk_it->second;
            } else {
                bgk = new SemanticBKI3f(
                            SemanticOcTreeNode::num_class,
                            SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
                bgk_arr.emplace(key, bgk);
            }
            bgk->addPoint(pos, label);
        }

#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: filled blocks: " << bgk_arr.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////

#pragma omp parallel
        {
#pragma omp single
            for (auto bgk_it = bgk_arr.cbegin(); bgk_it != bgk_arr.cend();  bgk_it++) {
#pragma omp task
                {
                    BlockHashKey key = bgk_it->first;
                    SemanticBKI3f *bgk = bgk_it->second;
                    Block *block;
#pragma omp critical
                    {
                        if (block_arr.find(key) == block_arr.end())
                            block_arr.emplace(key, new Block(hash_key_to_block(key)));
                        block = block_arr[key];
                    };
                    vector<float> xs;
                    auto leaf_it = block->begin_leaf();
                    // treat leaf as point for old points
                    point3f p = block->get_loc(leaf_it);
                    xs.push_back(p.x());
                    xs.push_back(p.y());
                    xs.push_back(p.z());

                    vector<vector<float>> ybars;
                    bgk->predict_csm(xs, ybars);

                    int j = 0;
                    leaf_it.get_node().update(ybars[j]);
                    //for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    //    SemanticOcTreeNode &node = leaf_it.get_node();

                    //    // Only need to update if kernel density total kernel density est > 0
                    //    node.update(ybars[j]);
                    //}

                }
            }
        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif

        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;
    }


    void SemanticBKIOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {
#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);
#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif
        // If pointcloud after max_range filtering is empty
        //  no need to do anything
        if (xy.size() == 0) {
            return;
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        // all blocks
        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
 
        std::unordered_map<BlockHashKey, SemanticBKI3f *> bgk_arr;
        for (auto point_it = xy.begin(); point_it != xy.end(); ++point_it) {
            point3f pos(point_it->x, point_it->y, point_it->z);
            auto label = point_it->label;
            BlockHashKey key = block_to_hash_key(pos);

            SemanticBKI3f *bgk;
            auto bgk_it = bgk_arr.find(key);
            if (bgk_it != bgk_arr.end()) {
                bgk = bgk_it->second;
            } else {
                bgk = new SemanticBKI3f(
                            SemanticOcTreeNode::num_class,
                            SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
                bgk_arr.emplace(key, bgk);
            }
            bgk->addPoint(pos, label);
        }

#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: filled blocks: " << bgk_arr.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////

#pragma omp parallel for schedule(dynamic)
        for (BlockHashKey key : blocks) {
            ExtendedBlock eblock = get_extended_block(key);
            std::vector<SemanticBKI3f*> neighbor_bkis;
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto bgk = bgk_arr.find(*block_it);
                if (bgk != bgk_arr.end()) {
                    neighbor_bkis.push_back(bgk->second);
                }
            }

            if (neighbor_bkis.empty()) {
                continue;
            }
            Block *block;
#pragma omp critical
            {
                if (block_arr.find(key) == block_arr.end()) {
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
                }
                block = block_arr[key];
            };
            vector<float> xs;
            // treat leaf as point for old points
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }
            //std::cout << "xs size: "<<xs.size() << std::endl;

            // for all bgk inference blocks in the extended block, do a prediction (for each test block?)
            for (auto bgk : neighbor_bkis) {

                // old predict
               	//vector<vector<float>> ybars;
                //bgk->predict(xs, ybars);
                //(*block)[block->get_node(0, 0, 0)].update(ybars[0]);

                // new predict
                auto predictions = bgk->new_predict(xs);
                (*block)[block->get_node(0, 0, 0)].update(predictions.array());
            }
        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif

        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;
    }

    void SemanticBKIOctoMap::insert_pointcloud_kdtree(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);

        // disable destructor, ownership is not managed by shared_ptr
        auto cloud_ptr = pcl::shared_ptr<semantic_bki::PCLPointCloud>(
            &xy, [](semantic_bki::PCLPointCloud *p) { (void)p; });
        pcl::KdTreeFLANN<semantic_bki::PCLPointType> kdtree;
        kdtree.setInputCloud(cloud_ptr);

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        // all blocks
        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        const SemanticBKI3f::Params params = {
            SemanticOcTreeNode::ell,
            SemanticOcTreeNode::sf2,
            SemanticOcTreeNode::num_class
        };

#pragma omp parallel for schedule(dynamic)
        for (BlockHashKey key : blocks) {
            PCLPointType block_center;
            auto center = hash_key_to_block(key);
            block_center.x = center.x();
            block_center.y = center.y();
            block_center.z = center.z();

            Block *block;
#pragma omp critical
            {
                if (block_arr.find(key) == block_arr.end()) {
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
                }
                block = block_arr[key];
            };

            std::vector<int> point_indices;
            std::vector<float> distances;
            kdtree.radiusSearch(block_center, params.max_distance, point_indices, distances);

            Eigen::VectorXi labels(point_indices.size());
            for (size_t i = 0; i < point_indices.size(); ++i) {
                labels(i) = xy[point_indices[i]].label;
            }
            Eigen::Map<Eigen::VectorXf> distances_v(distances.data(), distances.size());
            auto predictions = SemanticBKI3f::new_inner_predict(std::move(distances_v), labels, params);
            (*block)[block->get_node(0, 0, 0)].update(predictions.array());
        }
    }

    /**
     * Downsample input, discard hits greater than max_range, and add free samples by ray tracing
     */
    void SemanticBKIOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPPointCloud &xy) const {

        // downsample hits
        //PCLPointCloud sampled_hits;
        //downsample(cloud, sampled_hits, ds_resolution);
        const PCLPointCloud& sampled_hits = cloud;

        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;
        xy.clear();
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p(it->x, it->y, it->z);

            // create free samples from single beam
            PointCloud frees_n;
            beam_sample(p, origin, frees_n, free_resolution, max_range);

            // add free samples to list of all free samples
            for (auto p = frees_n.begin(); p != frees_n.end(); ++p) {
                PCLPointType p_free = PCLPointType();
                p_free.x = p->x();
                p_free.y = p->y();
                p_free.z = p->z();
                p_free.label = 0;
                frees.push_back(p_free);
                frees.width++;
            }

            if (max_range > 0) {
                double l = (p - origin).norm();
                if (l > max_range)
                    continue;
            }
            
            // copy hits into output xy
            xy.push_back(*it);

            // class mapping
            if (xy.back().label < m_class_mapping.size()) {
                if (m_class_mapping[xy.back().label] >=
                    static_cast<uint32_t>(SemanticOcTreeNode::num_class)) {
                    ROS_WARN_STREAM_THROTTLE(
                        0.1f,
                        "got a label "
                            << xy.back().label
                            << " which will be mapped out of classes range");
                }
                xy.back().label = m_class_mapping[xy.back().label];
            } else {
                ROS_WARN_STREAM_THROTTLE(
                    1.f, "label " << xy.back().label
                                  << " out of range of class mapping, mapped to "
                                     "num_class");
                xy.back().label = SemanticOcTreeNode::num_class;
            }
        }

        // downsample free samples and add to output
        //PCLPointCloud sampled_frees;    
        //downsample(frees, sampled_frees, ds_resolution);
        const PCLPointCloud& sampled_frees = frees;

        for (auto it = sampled_frees.begin(); it != sampled_frees.end(); ++it) {
            xy.push_back(*it);
        }
    }

    void SemanticBKIOctoMap::downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const {
        if (ds_resolution < 0) {
            out = in;
            return;
        }

        PCLPointCloud::Ptr pcl_in(new PCLPointCloud(in));

        pcl::VoxelGrid<PCLPointType> sor;
        sor.setInputCloud(pcl_in);
        sor.setLeafSize(ds_resolution, ds_resolution, ds_resolution);
        sor.filter(out);
    }

    void SemanticBKIOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
                                float free_resolution, float max_range) const {
        static std::mt19937 rand_gen;
        std::uniform_real_distribution<float> dist(0.0, free_resolution);

        frees.clear();

        float x0 = origin.x();
        float y0 = origin.y();
        float z0 = origin.z();

        float x = hit.x();
        float y = hit.y();
        float z = hit.z();

        float l = (float) sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

        float nx = (x - x0) / l;
        float ny = (y - y0) / l;
        float nz = (z - z0) / l;

        float d = dist(rand_gen);
        while (d < l && d < max_range) {
            frees.emplace_back(x0 + nx * d, y0 + ny * d, z0 + nz * d);
            d += free_resolution;
        }
    }

    /*
     * Compute bounding box of pointcloud
     * Precondition: cloud non-empty
     */
    void SemanticBKIOctoMap::bbox(const GPPointCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        assert(cloud.size() > 0);
        vector<float> x, y, z;
        for (auto it = cloud.begin(); it != cloud.end(); ++it) {
            x.push_back(it->x);
            y.push_back(it->y);
            z.push_back(it->z);
        }

        auto xlim = std::minmax_element(x.cbegin(), x.cend());
        auto ylim = std::minmax_element(y.cbegin(), y.cend());
        auto zlim = std::minmax_element(z.cbegin(), z.cend());

        lim_min.x() = *xlim.first;
        lim_min.y() = *ylim.first;
        lim_min.z() = *zlim.first;

        lim_max.x() = *xlim.second;
        lim_max.y() = *ylim.second;
        lim_max.z() = *zlim.second;
    }

    void SemanticBKIOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                       vector<BlockHashKey> &blocks) const {
        // with very conservative edges
        for (float x = lim_min.x() - block_size; x <= lim_max.x() + 2 * block_size; x += block_size) {
            for (float y = lim_min.y() - block_size; y <= lim_max.y() + 2 * block_size; y += block_size) {
                for (float z = lim_min.z() - block_size; z <= lim_max.z() + 2 * block_size; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         GPPointCloud &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         GPPointCloud &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, SemanticBKIOctoMap::search_callback, static_cast<void *>(&out));
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, SemanticBKIOctoMap::count_callback, NULL);
    }

    bool SemanticBKIOctoMap::count_callback(GPPointType *p, void *arg) {
        return false;
    }

    bool SemanticBKIOctoMap::search_callback(GPPointType *p, void *arg) {
        GPPointCloud *out = static_cast<GPPointCloud *>(arg);
        PCLPointType pcl_point;
        pcl_point.x = p->first.x();
        pcl_point.y = p->first.y();
        pcl_point.z = p->first.z();
        pcl_point.label = static_cast<uint32_t>(p->second);
        out->push_back(pcl_point);
        return true;
    }


    int SemanticBKIOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         GPPointCloud &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *SemanticBKIOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    SemanticOcTreeNode SemanticBKIOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
          return SemanticOcTreeNode();
        } else {
          return SemanticOcTreeNode(block->search(p));
        }
    }

    SemanticOcTreeNode SemanticBKIOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
