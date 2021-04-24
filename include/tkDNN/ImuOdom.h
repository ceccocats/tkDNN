#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */

#ifdef __linux__
#include <unistd.h>
#elif _WIN32
#define _USE_MATH_DEFINES
#include <math.h>
#endif 

#include <mutex>
#include <Eigen/Dense>
#include "utils.h"
#include "tkdnn.h"

namespace tk { namespace dnn {

/**
 * 
 * @author Francesco Gatti
 */
class ImuOdom {
        
    public:
        tk::dnn::Network *net = nullptr;

        // Network input dim
        tk::dnn::dataDim_t dim0;
        tk::dnn::dataDim_t dim1;
        tk::dnn::dataDim_t dim2;

        // Network output dim
        tk::dnn::dataDim_t odim0;
        tk::dnn::dataDim_t odim1;

        // input pointers
        dnnType *i0_d, *i1_d, *i2_d;
        // output pointers
        dnnType *o0_d, *o1_d;

        // output eigen CPU
        Eigen::MatrixXf deltaP, deltaQ;

        Eigen::MatrixXd odomPOS, odomEULER;
        Eigen::Matrix3d odomROT;
        Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();

        ImuOdom() {}

        virtual ~ImuOdom() {}

        /**
         * Method used for initialize the class
         * 
         * @return Success of the initialization
         */
        bool init(std::string layers_path) {

            dim0 = tk::dnn::dataDim_t(1, 4, 1, 100);
            dim1 = tk::dnn::dataDim_t(1, 3, 1, 100);
            dim2 = tk::dnn::dataDim_t(1, 3, 1, 100);

            checkCuda( cudaMalloc(&i0_d, dim0.tot()*sizeof(dnnType)) );
            checkCuda( cudaMalloc(&i1_d, dim1.tot()*sizeof(dnnType)) );
            checkCuda( cudaMalloc(&i2_d, dim2.tot()*sizeof(dnnType)) );

            std::string c0_bin = layers_path + "/conv1d_7.bin";
            std::string c1_bin = layers_path + "/conv1d_8.bin";
            std::string c2_bin = layers_path + "/conv1d_9.bin";
            std::string c3_bin = layers_path + "/conv1d_10.bin";
            std::string c4_bin = layers_path + "/conv1d_11.bin";
            std::string c5_bin = layers_path + "/conv1d_12.bin";
            std::string l0_bin = layers_path + "/bidirectional_3.bin";
            std::string l1_bin = layers_path + "/bidirectional_4.bin";
            std::string d0_bin = layers_path + "/dense_3.bin";
            std::string d1_bin = layers_path + "/dense_4.bin";

            net = new tk::dnn::Network(dim0);
            tk::dnn::Input   *x0   = new tk::dnn::Input  (net, dim0, i0_d);
            tk::dnn::Conv2d  *x0_0 = new tk::dnn::Conv2d (net, 128, 1, 11, 1, 1, 0, 0, c0_bin);
            tk::dnn::Conv2d  *x0_1 = new tk::dnn::Conv2d (net, 128, 1, 11, 1, 1, 0, 0, c1_bin);
            tk::dnn::Pooling *x0_2 = new tk::dnn::Pooling(net, 1, 3, 1, 3 ,0, 0, tk::dnn::tkdnnPoolingMode_t::POOLING_MAX);

            tk::dnn::Input   *x1   = new tk::dnn::Input  (net, dim1, i1_d);
            tk::dnn::Conv2d  *x1_0 = new tk::dnn::Conv2d (net, 128, 1, 11, 1, 1, 0, 0, c2_bin);
            tk::dnn::Conv2d  *x1_1 = new tk::dnn::Conv2d (net, 128, 1, 11, 1, 1, 0, 0, c3_bin);
            tk::dnn::Pooling *x1_2 = new tk::dnn::Pooling(net, 1, 3, 1, 3, 0, 0, tk::dnn::tkdnnPoolingMode_t::POOLING_MAX);

            tk::dnn::Input   *x2   = new tk::dnn::Input  (net, dim2, i2_d);
            tk::dnn::Conv2d  *x2_0 = new tk::dnn::Conv2d (net, 128, 1, 11, 1, 1, 0, 0, c4_bin);
            tk::dnn::Conv2d  *x2_1 = new tk::dnn::Conv2d (net, 128, 1, 11, 1, 1, 0, 0, c5_bin);
            tk::dnn::Pooling *x2_2 = new tk::dnn::Pooling(net, 1, 3, 1, 3, 0, 0, tk::dnn::tkdnnPoolingMode_t::POOLING_MAX);

            tk::dnn::Layer *concat_l[3] = { x0_2, x1_2, x2_2 };
            tk::dnn::Route *concat = new tk::dnn::Route(net, concat_l, 3);

            tk::dnn::LSTM *lstm0 = new tk::dnn::LSTM(net, 128, true, l0_bin);
            tk::dnn::LSTM *lstm1 = new tk::dnn::LSTM(net, 128, false, l1_bin);

            tk::dnn::Dense *d0 = new tk::dnn::Dense(net, 3, d0_bin);

            tk::dnn::Layer *lstm1_l[1] = { lstm1 };
            tk::dnn::Route *lstm1_link = new tk::dnn::Route(net, lstm1_l, 1);
            tk::dnn::Dense *d1 = new tk::dnn::Dense(net, 4, d1_bin);
            
            net->print();

            // output data
            o0_d = d0->dstData;
            o1_d = d1->dstData;
            odim0 = d0->output_dim;
            odim1 = d1->output_dim;

            deltaP.resize(odim0.tot(), 1);
            deltaQ.resize(odim1.tot(), 1);

            odomPOS = Eigen::MatrixXd::Zero(3, 1); 
            odomROT = Eigen::MatrixXd::Identity(3, 3);
            odomEULER = Eigen::MatrixXd::Zero(3, 1); 
            return true;
        }

        void close() {
            // TODO: dealloc :)
        }

        void update(dnnType *x0, dnnType *x1, dnnType *x2) {

            checkCuda( cudaMemcpy(i0_d, x0, dim0.tot()*sizeof(dnnType), cudaMemcpyHostToDevice) );
            checkCuda( cudaMemcpy(i1_d, x1, dim1.tot()*sizeof(dnnType), cudaMemcpyHostToDevice) );
            checkCuda( cudaMemcpy(i2_d, x2, dim2.tot()*sizeof(dnnType), cudaMemcpyHostToDevice) );

            // Inference
            tk::dnn::dataDim_t dim;
            net->infer(dim, nullptr);

            checkCuda( cudaMemcpy(deltaP.data(), o0_d, odim0.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );
            checkCuda( cudaMemcpy(deltaQ.data(), o1_d, odim1.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) ); 

            // compute odom
            Eigen::Quaterniond q;
            q.w() = deltaQ(0);
            q.x() = deltaQ(1);
            q.y() = deltaQ(2);
            q.z() = deltaQ(3);    
            odomPOS = odomPOS + odomROT*deltaP.cast<double>(); // V1
            //odomPOS = odomPOS + deltaP.cast<double>(); // V2
            odomROT = odomROT * q.normalized().toRotationMatrix();
            
            // compute Euler
            auto newEULER = odomROT.eulerAngles(0, 1, 2);
            for(int i=0; i<3; i++) {
                while( fabs(newEULER(i) - odomEULER(i)) > M_PI_2 )  {
                    newEULER(i) += newEULER(i) - odomEULER(i) > 0 ? -M_PI : +M_PI;
                    //std::cout<<newEULER(i)<<" "<<odomEULER(i)<<"\n";
                }
            }
            odomEULER = newEULER;

            // compose tf
            tf.matrix().block(0, 0, 3, 3) = odomROT.cast<float>();
            tf.matrix().block(0, 3, 3, 1) = odomPOS.cast<float>();
        }

};

}}
