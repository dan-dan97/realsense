#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

//#include <opencv2/opencv.hpp>
//
//#include <Eigen/Eigen>

#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseArray.h>
#include <std_msgs/String.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <geometry_msgs/PoseArray.h>

#define W 640
#define H 480

#define minDistToPlane 0.05
#define maxDistToPlane 3
#define PLANE_OFFSET 0.004

#define DEBUG
#define ALWAYS_WORK

bool enableCV = 0;

template <typename type>
type sqr(type arg){
    return arg * arg;
}

namespace pcl{
    template <typename type>
    boost::shared_ptr<type> make_shared(type& arg){
        return boost::make_shared<type>(arg);
    }
}

void getRealsenseXYZRGBCloud(rs2::pipeline& sensorStream, rs2::frameset& frames, pcl::PointCloud<pcl::PointXYZ>& pointCloudXYZ, pcl::PointCloud<pcl::PointXYZRGB>& pointCloudXYZRGB){
    pointCloudXYZRGB.clear();

    rs2::depth_frame depth_frame = frames.first_or_default(RS2_STREAM_DEPTH);
    rs2::video_frame color_frame = frames.first_or_default(RS2_STREAM_COLOR);

    if(!(depth_frame && color_frame)) return;

    static rs2_extrinsics depthToColor = sensorStream.get_active_profile().get_stream(RS2_STREAM_DEPTH).get_extrinsics_to(sensorStream.get_active_profile().get_stream(RS2_STREAM_COLOR));
    static rs2_intrinsics colorIntrin = sensorStream.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

    float depthPoint[3];
    float colorPoint[3];
    float colorPixel[2];

    for(pcl::PointXYZ pointXYZ : pointCloudXYZ){
        depthPoint[0] = pointXYZ.x;
        depthPoint[1] = pointXYZ.y;
        depthPoint[2] = pointXYZ.z;
        rs2_transform_point_to_point(colorPoint, &depthToColor, depthPoint);
        rs2_project_point_to_pixel(colorPixel, &colorIntrin, colorPoint);

        int colorPixelX = round(colorPixel[0]);
        int colorPixelY = round(colorPixel[1]);
        if(!(colorPixelX >= 0 && colorPixelX < color_frame.get_width() && colorPixelY >= 0 && colorPixelY < color_frame.get_height())) continue;

        uint8_t* color = (uint8_t*)color_frame.get_data() + colorPixelY * color_frame.get_stride_in_bytes() + colorPixelX * color_frame.get_bytes_per_pixel();
        pcl::PointXYZRGB pointXYZRGB;
        pcl::copyPoint(pointXYZ, pointXYZRGB);
        pointXYZRGB.r = color[2];
        pointXYZRGB.g = color[1];
        pointXYZRGB.b = color[0];
        pointCloudXYZRGB.push_back(pointXYZRGB);
    }
}

void stateCallback(const std_msgs::String::ConstPtr& msg)
{
    enableCV = (msg->data == "Search enable");
}

int main (int argc, char** argv) {

    ros::init(argc, argv, "cv3d_hangover");
    ros::NodeHandle node;

    ros::Rate rate(100);

    ros::Publisher objectsPublisher = node.advertise<geometry_msgs::PoseArray>("/detected_objects", 10);
    ros::Subscriber stateSubscriber = node.subscribe("/drone_state", 1, stateCallback);

    tf::TransformListener listener;

    rs2::config cfg;
    cfg.disable_all_streams();
    cfg.enable_stream(RS2_STREAM_COLOR, 0, W, H, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, W, H, RS2_FORMAT_Z16, 60);
    //cfg.enable_stream(RS2_STREAM_INFRARED, W, H, RS2_FORMAT_Y16, 60);

    rs2::pipeline pipe;
    rs2::pipeline_profile selection = pipe.start(cfg);
    rs2::depth_sensor depthSensor = pipe.get_active_profile().get_device().first<rs2::depth_sensor>();
    rs2_intrinsics colorIntrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    rs2_intrinsics depthIntrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
    rs2_extrinsics depthToColorExtrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).get_extrinsics_to(pipe.get_active_profile().get_stream(RS2_STREAM_COLOR));

    depthSensor.set_option(RS2_OPTION_CONFIDENCE_THRESHOLD, depthSensor.get_option_range(RS2_OPTION_CONFIDENCE_THRESHOLD).max);
    depthSensor.set_option(RS2_OPTION_ACCURACY, depthSensor.get_option_range(RS2_OPTION_ACCURACY).max);
    depthSensor.set_option(RS2_OPTION_MOTION_RANGE, 120);

#ifdef DEBUG
    pcl::visualization::PCLVisualizer cloudViewer("Cloud viewer");
    cloudViewer.setBackgroundColor(0, 0, 0);
    cloudViewer.addCoordinateSystem(0.01);
    cloudViewer.initCameraParameters();
    cloudViewer.setCameraPosition(0, 0, 0, 0, -1, -1);
#endif

    bool mode = 0;

    geometry_msgs::PoseArray poseArray;
    poseArray.header.seq = 0;

    while (ros::ok())
    {
        boost::posix_time::ptime t0 = boost::posix_time::microsec_clock::local_time();

        do{

#ifndef ALWAYS_WORK
            if(!enableCV) break;
#endif

            tf::StampedTransform globalCameraTF;
#ifndef ALWAYS_WORK
            bool tfNotFound = 0;
            try { listener.lookupTransform("/camera_link", "/aruco_map", ros::Time(0), globalCameraTF); }
            catch (tf::TransformException &ex) { tfNotFound = 1; }
            if(tfNotFound) break;
#endif

#ifdef DEBUG
            if(cloudViewer.wasStopped())
                break;
            cloudViewer.removeAllPointClouds();
            cloudViewer.removeAllShapes();
#endif
            rs2::frameset frames = pipe.wait_for_frames();
            rs2::depth_frame depth_frame = frames.first_or_default(RS2_STREAM_DEPTH);
            rs2::video_frame color_frame = frames.first_or_default(RS2_STREAM_COLOR);
            //rs2::video_frame infrared_frame = frames.first_or_default(RS2_STREAM_INFRARED);

            if (!(depth_frame && color_frame)) break;

            static pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorImagePointCloud(
                    new pcl::PointCloud<pcl::PointXYZRGB>(W, H));

            static pcl::PointCloud<pcl::PointXYZ>::Ptr sourcePointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
            static pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourcePointCloudXYZRGB(
                    new pcl::PointCloud<pcl::PointXYZRGB>);
            static pcl::PointCloud<pcl::PointXYZ>::Ptr filteredPointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
            static pcl::PointCloud<pcl::PointXYZ>::Ptr filteredObjPointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
            static pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredPointCloudXYZRGB(
                    new pcl::PointCloud<pcl::PointXYZRGB>);

            static pcl::PointCloud<pcl::PointXYZ>::Ptr outOfPlanePointCloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);

            static std::vector<pcl::PointCloud<pcl::PointXYZ>> objectsPointCloudsXYZ;
            static std::vector<tf::StampedTransform> objectsCenter;

            static std::vector<pcl::PointXYZ> objectsCenterPoint;

            //static std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Quaternionf>> objectsBoundaryBox;

            static pcl::VoxelGrid<pcl::PointXYZ> voxelGridFilter;
            static pcl::SACSegmentation<pcl::PointXYZ> sacSegmentation;
            static pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclideanClusterExtraction;
            static pcl::MomentOfInertiaEstimation<pcl::PointXYZ> featureExtractor;

            static pcl::ModelCoefficients::Ptr workPlaneCoefficientsPCL(new pcl::ModelCoefficients);
            static pcl::PointIndices::Ptr pointIndices(new pcl::PointIndices);
            static pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);
            static std::vector<pcl::PointIndices> clusterPointsIndices;

            sourcePointCloudXYZ->clear();
            for (pcl::PointXYZRGB &point : *colorImagePointCloud) {
                point.x = 0;
                point.y = 0;
                point.z = 0;
            }

            for (int ix = 0; ix < depth_frame.get_width(); ix++)
                for (int iy = 0; iy < depth_frame.get_height(); iy++) {
                    float dist = depth_frame.get_distance(ix, iy);
                    static float depthPoint[3];
                    static float colorPoint[3];
                    static float depthPixel[2];
                    static float colorPixel[2];
                    depthPixel[0] = ix;
                    depthPixel[1] = iy;

                    if (dist > 0) {
                        rs2_deproject_pixel_to_point(depthPoint, &depthIntrinsics, depthPixel, dist);
                        rs2_transform_point_to_point(colorPoint, &depthToColorExtrinsics, depthPoint);
                        rs2_project_point_to_pixel(colorPixel, &colorIntrinsics, colorPoint);

                        sourcePointCloudXYZ->push_back(pcl::PointXYZ(depthPoint[0], depthPoint[1], depthPoint[2]));

                        int colorPixelX = round(colorPixel[0]);
                        int colorPixelY = round(colorPixel[1]);
                        if (!(colorPixelX >= 0 && colorPixelX < color_frame.get_width() && colorPixelY >= 0 &&
                              colorPixelY < color_frame.get_height()))
                            continue;

                        uint8_t *color =
                                (uint8_t *) color_frame.get_data() +
                                colorPixelY * color_frame.get_stride_in_bytes() +
                                colorPixelX * color_frame.get_bytes_per_pixel();

                        colorImagePointCloud->at(colorPixelX, colorPixelY).x = depthPoint[0];
                        colorImagePointCloud->at(colorPixelX, colorPixelY).y = depthPoint[1];
                        colorImagePointCloud->at(colorPixelX, colorPixelY).z = depthPoint[2];
                        colorImagePointCloud->at(colorPixelX, colorPixelY).r = color[2];
                        colorImagePointCloud->at(colorPixelX, colorPixelY).g = color[1];
                        colorImagePointCloud->at(colorPixelX, colorPixelY).b = color[0];
                    }
                }

            for (int ix = 0; ix < colorImagePointCloud->width; ix++)
                for (int iy = 0; iy < colorImagePointCloud->height; iy++) {
                    pcl::PointXYZRGB &point = colorImagePointCloud->at(ix, iy);
                    if (point.z == 0) {
                        uint8_t *color =
                                (uint8_t *) color_frame.get_data() + iy * color_frame.get_stride_in_bytes() +
                                ix * color_frame.get_bytes_per_pixel();
                        point.r = color[2];
                        point.g = color[1];
                        point.b = color[0];
                    }
                }

            getRealsenseXYZRGBCloud(pipe, frames, *sourcePointCloudXYZ, *sourcePointCloudXYZRGB);

            voxelGridFilter.setInputCloud(sourcePointCloudXYZ);
            float voxelGridSize = 0.01; //0.01f;
            voxelGridFilter.setLeafSize(voxelGridSize, voxelGridSize, voxelGridSize);
            voxelGridFilter.filter(*filteredPointCloudXYZ);

            sacSegmentation.setOptimizeCoefficients(true);
            sacSegmentation.setModelType(pcl::SACMODEL_PLANE);
            sacSegmentation.setMethodType(pcl::SAC_RANSAC);
            sacSegmentation.setMaxIterations(500);
            sacSegmentation.setDistanceThreshold(0.05);
            sacSegmentation.setInputCloud(filteredPointCloudXYZ);
            sacSegmentation.segment(*pointIndices, *workPlaneCoefficientsPCL);

            if (pointIndices->indices.empty())break;

            std::pair<Eigen::Vector3d, double> planeCoefficients;
            planeCoefficients.first
                    << workPlaneCoefficientsPCL->values[0], workPlaneCoefficientsPCL->values[1], workPlaneCoefficientsPCL->values[2];
            planeCoefficients.second = workPlaneCoefficientsPCL->values[3];
            planeCoefficients.second /= planeCoefficients.first.norm();
            planeCoefficients.first /= planeCoefficients.first.norm();
            if (planeCoefficients.first.z() > 0) {
                planeCoefficients.first = -planeCoefficients.first;
                planeCoefficients.second = -planeCoefficients.second;
            }
            if (planeCoefficients.first.z() == 0) break;

            (*filteredObjPointCloudXYZ).clear();
            for (pcl::PointXYZ &oneSourcePoint : *filteredPointCloudXYZ) {
                double distanceToPlane = (planeCoefficients.first.x() * oneSourcePoint.x) +
                                         (planeCoefficients.first.y() * oneSourcePoint.y) +
                                         (planeCoefficients.first.z() * oneSourcePoint.z) +
                                         planeCoefficients.second;
                if (distanceToPlane > minDistToPlane)
                    filteredObjPointCloudXYZ->push_back(oneSourcePoint);
            }

            if(filteredObjPointCloudXYZ->empty()) break;

            kdTree->setInputCloud(filteredObjPointCloudXYZ);
            clusterPointsIndices.clear();
            euclideanClusterExtraction.setClusterTolerance(0.25);//dist between the obj
            euclideanClusterExtraction.setMinClusterSize(filteredObjPointCloudXYZ->size() * 0.2);
            euclideanClusterExtraction.setMaxClusterSize(filteredObjPointCloudXYZ->size());
            euclideanClusterExtraction.setSearchMethod(kdTree);
            euclideanClusterExtraction.setInputCloud(filteredObjPointCloudXYZ);
            euclideanClusterExtraction.extract(clusterPointsIndices);

            if (clusterPointsIndices.empty()) break;

            objectsPointCloudsXYZ.clear();

            for (pcl::PointIndices &currentClusterPointsIndices : clusterPointsIndices) {
                objectsPointCloudsXYZ.resize(objectsPointCloudsXYZ.size() + 1);
                for (int indice : currentClusterPointsIndices.indices)
                    objectsPointCloudsXYZ[objectsPointCloudsXYZ.size() - 1].push_back(filteredObjPointCloudXYZ->at(indice));
            }

            tf::Transform resultTF;
            int maxSizeObject = -1;

            objectsCenterPoint.resize(objectsPointCloudsXYZ.size());
            for(int i = 0; i < objectsPointCloudsXYZ.size(); i++){

                Eigen::Vector3f massCenter;
                featureExtractor.setInputCloud(pcl::make_shared(objectsPointCloudsXYZ[i]));
                featureExtractor.compute();
                featureExtractor.getMassCenter(massCenter);

                objectsCenterPoint[i].x = massCenter.x();
                objectsCenterPoint[i].y = massCenter.y();
                objectsCenterPoint[i].z = massCenter.z();

                tf::Transform objectTF = globalCameraTF * tf::Transform(tf::Quaternion(0, 0, 0, 1), tf::Vector3(massCenter.x(), massCenter.y(), massCenter.z()));

                if(objectsPointCloudsXYZ[i].size() > maxSizeObject){
                    maxSizeObject = objectsPointCloudsXYZ[i].size();
                    resultTF = objectTF;
                }
            }

            std::cout << "Objects: " << objectsPointCloudsXYZ.size() << std::endl;


#ifndef ALWAYS_WORK
            poseArray.header.frame_id = "/object";
            poseArray.header.seq++;
            poseArray.header.stamp = ros::Time::now();
            poseArray.poses.resize(1);
            poseArray.poses[0].orientation.x = 0;
            poseArray.poses[0].orientation.y = 0;
            poseArray.poses[0].orientation.z = 0;
            poseArray.poses[0].orientation.w = 1;
            poseArray.poses[0].position.x = resultTF.getOrigin().x();
            poseArray.poses[0].position.y = resultTF.getOrigin().y();

            objectsPublisher.publish(poseArray);
#endif


#ifdef DEBUG
            static std::vector<pcl::RGB> colors(6);
            colors[0].r = 255;
            colors[0].g = 0;
            colors[0].b = 0;

            colors[1].r = 0;
            colors[1].g = 255;
            colors[1].b = 0;

            colors[2].r = 0;
            colors[2].g = 0;
            colors[2].b = 255;

            colors[3].r = 255;
            colors[3].g = 255;
            colors[3].b = 0;

            colors[4].r = 255;
            colors[4].g = 0;
            colors[4].b = 255;

            colors[5].r = 0;
            colors[5].g = 255;
            colors[5].b = 255;

            typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorizerXYZ;
            typedef pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> colorizerXYZRGB;
            static pcl::PointCloud<pcl::PointXYZRGB>::Ptr vizualizedPointCloudXYZRGB(
                    new pcl::PointCloud<pcl::PointXYZRGB>);


//                    getRealsenseXYZRGBCloud(pipe, frames, *filteredObjPointCloudXYZ, *vizualizedPointCloudXYZRGB);
//                    cloudViewer.addPointCloud<pcl::PointXYZRGB>(vizualizedPointCloudXYZRGB, colorizerXYZRGB(vizualizedPointCloudXYZRGB), "point cloud");
//                    cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "point cloud");


                for (int i = 0; i < objectsPointCloudsXYZ.size(); i++) {
                    static pcl::PointCloud<pcl::PointXYZRGB>::Ptr vizualizedPointCloudXYZRGB(
                            new pcl::PointCloud<pcl::PointXYZRGB>);
                    getRealsenseXYZRGBCloud(pipe, frames, objectsPointCloudsXYZ[i], *vizualizedPointCloudXYZRGB);
                    cloudViewer.addPointCloud<pcl::PointXYZRGB>(vizualizedPointCloudXYZRGB,
                                                                colorizerXYZRGB(vizualizedPointCloudXYZRGB),
                                                                "object" + std::to_string(i));
                    cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,
                                                                 "object" + std::to_string(i));

                    pcl::RGB color = colors[i % colors.size()];

                    pcl::PointCloud<pcl::PointXYZ> centerPoint(1, 1, objectsCenterPoint[i]);
                    cloudViewer.addPointCloud<pcl::PointXYZ>(centerPoint.makeShared(), colorizerXYZ(centerPoint.makeShared(), color.r, color.g, color.b), "center" + std::to_string(i));
                    cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "center" + std::to_string(i));
                }

                //cloudViewer.addPlane(*workPlaneCoefficientsPCL, "Work plane");
//                    for (int i = 0; i < objectsPointCloudsXYZ.size(); i++) {
//                        static pcl::PointCloud<pcl::PointXYZRGB>::Ptr vizualizedPointCloudXYZRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
//                        getRealsenseXYZRGBCloud(pipe, frames, objectsPointCloudsXYZ[i], *vizualizedPointCloudXYZRGB);
//                        cloudViewer.addPointCloud<pcl::PointXYZRGB>(vizualizedPointCloudXYZRGB, colorizerXYZRGB(vizualizedPointCloudXYZRGB), "object" + std::to_string(i));
//                        cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "object" + std::to_string(i));
//
//                        pcl::RGB color = colors[i % colors.size()];
//
//                        cloudViewer.addPointCloud<pcl::PointXYZ>(contoursPointCloudsXYZ[i].makeShared(), colorizerXYZ(contoursPointCloudsXYZ[i].makeShared(), color.r, color.g, color.b), "contour" + std::to_string(i));
//                        cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "contour" + std::to_string(i));
//
//                        pcl::PointCloud<pcl::PointXYZ> centerPoint(1, 1, objectsCenter[i]);
//                        pcl::PointCloud<pcl::PointXYZ> centerPoint(1, 1, pcl::PointXYZ(std::get<0>(objectsBoundaryBox[i]).x(), std::get<0>(objectsBoundaryBox[i]).y(), std::get<0>(objectsBoundaryBox[i]).z()));
//                        cloudViewer.addPointCloud<pcl::PointXYZ>(centerPoint.makeShared(), colorizerXYZ(centerPoint.makeShared(), color.r, color.g, color.b), "center" + std::to_string(i));
//                        cloudViewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "center" + std::to_string(i));
//
//                        cloudViewer.addCube(std::get<0>(objectsBoundaryBox[i]), std::get<2>(objectsBoundaryBox[i]), std::get<1>(objectsBoundaryBox[i]).x(), std::get<1>(objectsBoundaryBox[i]).y(), std::get<1>(objectsBoundaryBox[i]).z(), "box" + std::to_string(i));
//                        cloudViewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r / 255.0, color.g / 255.0, color.b / 255.0, "box" + std::to_string(i));
//                        cloudViewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "box" + std::to_string(i));
//                    }
            //}
#endif
//                std::cout << "Iteration time: "
//                          << (boost::posix_time::microsec_clock::local_time() - t0).total_milliseconds() << std::endl;

#ifndef ALWAYS_WORK
            ROS_INFO("Object detected: x=%f\ty=%f\n", poseArray.poses[0].position.x, poseArray.poses[0].position.y);
#endif


        } while(false);

#ifdef DEBUG
        cloudViewer.spinOnce();
#endif

        ros::spinOnce();
        rate.sleep();

        //cv::waitKey(1);
    }

    ros::Duration(1).sleep();

#ifdef DEBUG
    cloudViewer.close();
#endif
    return 0;
}