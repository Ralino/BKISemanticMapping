#pragma once

#include "semantic_bki/mapping/bkioctomap.h"

static const float M_PI_f = M_PI;

namespace semantic_bki {

	/*
     * @brief Bayesian Generalized Kernel Inference on Bernoulli distribution
     * @param dim dimension of data (2, 3, etc.)
     * @param T data type (float, double, etc.)
     * @ref Nonparametric Bayesian inference on multivariate exponential families
     */
    template<int dim, typename T>
    class SemanticBKInference {
    public:
        /// Eigen matrix type for training and test data and kernel
        using MatrixXType = Eigen::Matrix<T, -1, dim, Eigen::RowMajor>;
        using MatrixKType = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
        using MatrixDKType = Eigen::Matrix<T, -1, 1>;
        using MatrixYType = Eigen::Matrix<T, -1, 1>;

        SemanticBKInference(int nc, T sf2, T ell) : nc(nc), sf2(sf2), ell(ell) { }

        void addPoint(point3f p, float label) {
            this->x_vec.push_back(p.x());
            this->x_vec.push_back(p.y());
            this->x_vec.push_back(p.z());
            this->y_vec.push_back(label);
        }

        struct Params {
            float max_distance; // l in paper
            float variance;     // sigma_0 in paper
            int num_classes;
        };

        static Eigen::VectorXf new_dist(const Eigen::MatrixX3f& points, const Eigen::Vector3f origin) {
            return (points.rowwise() - origin.transpose()).rowwise().norm();
        }

        static Eigen::VectorXf new_inner_predict(Eigen::VectorXf &&distances,
                const Eigen::VectorXi &labels, const Params& params) {
            assert(distances.size() == labels.size());

            distances /= params.max_distance;
            const Eigen::RowVectorXf kernel =
                ((2.f + (distances * (2.f * M_PI_f)).array().cos()) *
                     (1.f - distances.array()) * (params.variance / 3.f) +
                 (distances * (2.f * M_PI_f)).array().sin() *
                     (params.variance / (2.f * M_PI_f)))
                    .transpose();
            Eigen::VectorXf class_probs(params.num_classes);

            Eigen::VectorXf zero_vec = Eigen::VectorXf::Zero(labels.rows());
            for (int k = 0; k < params.num_classes; ++k) {
                decltype(zero_vec) class_filter = (labels.array() == k).select(1.f, zero_vec);
                const auto prob = kernel * class_filter;
                assert(prob.rows() == 1 && prob.cols() == 1);
                class_probs(k) = prob(0,0);
            }

            return class_probs;
        }

        Eigen::VectorXf new_predict(const std::vector<float>& origin) {
            const Eigen::Map<const MatrixXType> points(x_vec.data(), x_vec.size() / dim, dim);
            Eigen::VectorXi labels(y_vec.size());
            for (int k = 0; k < y_vec.size(); ++k) {
                labels(k) = static_cast<int>(y_vec[k]);
            }
            Params params{ell, sf2, nc};
            auto all_distances = new_dist(points, {origin[0], origin[1], origin[2]});
            for (int i = 0; i < all_distances.rows(); ++i) {
                if (all_distances(i) > params.max_distance) {
                    all_distances(i) = params.max_distance;
                }
            }
            return new_inner_predict(std::move(all_distances), labels, params);
        }


        void predict(const std::vector<T> &xs, std::vector<std::vector<T>> &ybars) const {
            assert(xs.size() % dim == 0);
            MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);
            const Eigen::Map<const MatrixXType> x(x_vec.data(), x_vec.size() / dim, dim);
            MatrixKType Ks;

            // _xs: leafs, x: new laser points
            covSparse(_xs, x, Ks);

            ybars.resize(_xs.rows());
            for (int r = 0; r < _xs.rows(); ++r)
                ybars[r].resize(nc);

            MatrixYType _y_vec = Eigen::Map<const MatrixYType>(y_vec.data(), y_vec.size(), 1);
            for (int k = 0; k < nc; ++k) {
                for (int i = 0; i < y_vec.size(); ++i) {
                    if (y_vec[i] == k)
                        _y_vec(i, 0) = 1;
                    else
                        _y_vec(i, 0) = 0;
                }

                MatrixYType _ybar;
                _ybar = (Ks * _y_vec);

                for (int r = 0; r < _ybar.rows(); ++r)
                    ybars[r][k] = _ybar(r, 0);
            }
        }

        void predict_csm(const std::vector<T> &xs, std::vector<std::vector<T>> &ybars) {
            assert(xs.size() % dim == 0);
            MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);
            const Eigen::Map<const MatrixXType> x(x_vec.data(), x_vec.size() / dim, dim);
            MatrixKType Ks;

            covCountingSensorModel(_xs, x, Ks);

            ybars.resize(_xs.rows());
            for (int r = 0; r < _xs.rows(); ++r)
                ybars[r].resize(nc);

            MatrixYType _y_vec = Eigen::Map<const MatrixYType>(y_vec.data(), y_vec.size(), 1);
            for (int k = 0; k < nc; ++k) {
                for (int i = 0; i < y_vec.size(); ++i) {
                    if (y_vec[i] == k)
                        _y_vec(i, 0) = 1;
                    else
                        _y_vec(i, 0) = 0;
                }

                MatrixYType _ybar;
                _ybar = (Ks * _y_vec);

                for (int r = 0; r < _ybar.rows(); ++r)
                    ybars[r][k] = _ybar(r, 0);
            }
        }


    private:
        /*
         * @brief Compute Euclid distances between two vectors.
         * @param x input vector
         * @param z input vecotr
         * @return d distance matrix
         */
        void dist(const MatrixXType &x, const MatrixXType &z, MatrixKType &d) const {
            d = MatrixKType::Zero(x.rows(), z.rows());
            for (int i = 0; i < x.rows(); ++i) {
                d.row(i) = (z.rowwise() - x.row(i)).rowwise().norm();
            }
        }

        /*
         * @brief Matern3 kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         */
        void covMaterniso3(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            dist(1.73205 / ell * x, 1.73205 / ell * z, Kxz);
            Kxz = ((1 + Kxz.array()) * exp(-Kxz.array())).matrix() * sf2;
        }

        /*
         * @brief Sparse kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         * @ref A sparse covariance function for exact gaussian process inference in large datasets.
         */
        void covSparse(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            dist(x / ell, z / ell, Kxz);
            Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
                  (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * sf2;

            // Clean up for values with distance outside length scale
            // Possible because Kxz <= 0 when dist >= ell
            for (int i = 0; i < Kxz.rows(); ++i)
            {
                for (int j = 0; j < Kxz.cols(); ++j) {
                    if (Kxz(i,j) < 0.0) {
                        Kxz(i,j) = 0.0f;
                    }
                }
            }
        }

        void covCountingSensorModel(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
          Kxz = MatrixKType::Ones(x.rows(), z.rows());
        }

        T sf2;    // signal variance
        T ell;    // length-scale
        int nc;   // number of classes

        std::vector<T> x_vec;
        std::vector<T> y_vec;
    };

    typedef SemanticBKInference<3, float> SemanticBKI3f;

}
