 /*
  * OpenCV Example using ROS and CPP
  */

 // Include the ROS library
 #include <ros/ros.h>

 // Include opencv2
 #include <opencv2/core/mat.hpp>
 #include <opencv2/highgui.hpp>
 #include <opencv2/imgproc.hpp>


 // Include CvBridge, Image Transport, Image msg
 #include <image_transport/image_transport.h>
 #include <cv_bridge/cv_bridge.h>
 #include <sensor_msgs/image_encodings.h>
 #include <sensor_msgs/PointCloud2.h>
 #include <geometry_msgs/Point.h>
 #include <geometry_msgs/PoseStamped.h>
 #include <geometry_msgs/Pose.h>

// Include tf2 for transformation
 #include <tf2_ros/buffer.h>
 #include <tf2_ros/transform_listener.h>
 #include <tf2_geometry_msgs/tf2_geometry_msgs.h>

 #include "opencv_services/box_and_target_position.h"
 #include "custom_msg/custom.h"

 // Topics
 static const std::string IMAGE_TOPIC = "/camera1/rgb/image_raw";
 static const std::string POINT_CLOUD2_TOPIC = "/camera1/depth/points";

 // Publisher
 ros::Publisher pub;

tf2_ros::Buffer tf_buffer;

const std::string from_frame = "camera_depth_optical_frame";
const std::string to_frame = "base_link";

cv::Mat camera_image;

cv::Point2f box_centroid;
cv::Point2f target_centroid;
cv::Point2f object_1_centroid;
cv::Point2f object_2_centroid;
cv::Point2f object_3_centroid;
cv::Point2f object_4_centroid;

float object_1_orientation;
float object_2_orientation;
float object_3_orientation;
float object_4_orientation;

int object_1_class;
int object_2_class;
int object_3_class;
int object_4_class;

geometry_msgs::Point object_1_position_base_frame;
geometry_msgs::Point object_2_position_base_frame;
geometry_msgs::Point object_3_position_base_frame;
geometry_msgs::Point object_4_position_base_frame;

cv::Point2f search_centroid_in_area(std::vector<cv::Point2f> centroid_vector, cv::Rect area) {
  float sum_x = 0.0;
  float sum_y = 0.0;
  int number_of_centroids_in_area = 0;
  
  for( int i = 0; i<centroid_vector.size(); i++) {
    if(centroid_vector[i].inside(area)) {
      sum_x += centroid_vector[i].x;
      sum_y += centroid_vector[i].y;
      number_of_centroids_in_area++;
    }
  }
  cv::Point2f extracted_point(sum_x/number_of_centroids_in_area, sum_y/number_of_centroids_in_area);
  return extracted_point;
}

cv::Mat apply_cv_algorithms(cv::Mat camera_image) {
  // convert the image to grayscale format
  cv::Mat img_gray;
  cv::cvtColor(camera_image, img_gray, cv::COLOR_BGR2GRAY);

  cv::Mat canny_output;
  cv::Canny(img_gray,canny_output,10,350);

  return canny_output;
}

std::vector<cv::Point2f> extract_centroids(cv::Mat canny_output) {

  // detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
   std::vector<std::vector<cv::Point>> contours;
   std::vector<cv::Vec4i> hierarchy;
   cv::findContours(canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

  // get the moments
  std::vector<cv::Moments> mu(contours.size());
  for( int i = 0; i<contours.size(); i++ )
  { mu[i] = cv::moments( contours[i], false ); }
  
  // get the centroid of figures.
  std::vector<cv::Point2f> centroids(contours.size());
  for( int i = 0; i<contours.size(); i++) {
    float centroid_x = mu[i].m10/mu[i].m00;
    float centroid_y = mu[i].m01/mu[i].m00;
    centroids[i] = cv::Point2f(centroid_x, centroid_y);
  }

    // draw contours
  cv::Mat drawing(canny_output.size(), CV_8UC3, cv::Scalar(255,255,255));

  for( int i = 0; i<contours.size(); i++ )
  {
  cv::Scalar color = cv::Scalar(167,151,0); // B G R values
  cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
  cv::circle( drawing, centroids[i], 4, color, -1, 8, 0 );
  }

   // show the resuling image
  cv::namedWindow( "Extracted centroids", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Extracted centroids", drawing );
  cv::waitKey(3);

  return centroids;
}

void image_cb(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  float image_size_y = cv_ptr->image.rows;
  float image_size_x = cv_ptr->image.cols;

  cv::Mat canny_output = apply_cv_algorithms(cv_ptr->image);

  std::vector<cv::Point2f> centroids = extract_centroids(canny_output);

  //get box location in 2d image
  cv::Rect box_search_area((image_size_x/2), 0, (image_size_x/2), 255);
  box_centroid = search_centroid_in_area(centroids, box_search_area);

  //get plate location in 2d image
  cv::Rect target_search_area(0, 0, (image_size_x/2), 255);
  target_centroid = search_centroid_in_area(centroids, target_search_area);

}


geometry_msgs::Point pixel_to_3d_point(const sensor_msgs::PointCloud2 pCloud, const int u, const int v)
{
  // get width and height of 2D point cloud data
  int width = pCloud.width;
  int height = pCloud.height;

  // Convert from u (column / width), v (row/height) to position in array
  // where X,Y,Z data starts
  int arrayPosition = v*pCloud.row_step + u*pCloud.point_step;

  // compute position in array where x,y,z data start
  int arrayPosX = arrayPosition + pCloud.fields[0].offset; // X has an offset of 0
  int arrayPosY = arrayPosition + pCloud.fields[1].offset; // Y has an offset of 4
  int arrayPosZ = arrayPosition + pCloud.fields[2].offset; // Z has an offset of 8

  float X = 0.0;
  float Y = 0.0;
  float Z = 0.0;

  memcpy(&X, &pCloud.data[arrayPosX], sizeof(float));
  memcpy(&Y, &pCloud.data[arrayPosY], sizeof(float));
  memcpy(&Z, &pCloud.data[arrayPosZ], sizeof(float));

  geometry_msgs::Point p;
  p.x = X;
  p.y = Y;
  p.z = Z;

  return p;
}

geometry_msgs::Point transform_between_frames(geometry_msgs::Point p, const std::string from_frame, const std::string to_frame) {
    
  geometry_msgs::PoseStamped input_pose_stamped;
  input_pose_stamped.pose.position = p;
  input_pose_stamped.header.frame_id = from_frame;
  input_pose_stamped.header.stamp = ros::Time::now();

  geometry_msgs::PoseStamped output_pose_stamped = tf_buffer.transform(input_pose_stamped, to_frame, ros::Duration(1));
  return output_pose_stamped.pose.position;
}

void point_cloud_cb(const sensor_msgs::PointCloud2 pCloud) {

  geometry_msgs::Point object_1_camera_frame;
  object_1_camera_frame = pixel_to_3d_point(pCloud, object_1_centroid.x, object_1_centroid.y);

  geometry_msgs::Point object_2_camera_frame;
  object_2_camera_frame = pixel_to_3d_point(pCloud, object_2_centroid.x, object_2_centroid.y);

  geometry_msgs::Point object_3_camera_frame;
  object_3_camera_frame = pixel_to_3d_point(pCloud, object_3_centroid.x, object_3_centroid.y);

  geometry_msgs::Point object_4_camera_frame;
  object_4_camera_frame = pixel_to_3d_point(pCloud, object_4_centroid.x, object_4_centroid.y);
    
  object_1_position_base_frame = transform_between_frames(object_1_camera_frame, from_frame, to_frame);
  object_2_position_base_frame = transform_between_frames(object_2_camera_frame, from_frame, to_frame);
  object_3_position_base_frame = transform_between_frames(object_3_camera_frame, from_frame, to_frame);
  object_4_position_base_frame = transform_between_frames(object_4_camera_frame, from_frame, to_frame);

  ROS_INFO_STREAM("Object 1 position base frame: x " << object_1_position_base_frame.x << " y " << object_1_position_base_frame.y << " orientation " << object_1_orientation);
  ROS_INFO_STREAM("Object 2 position base frame: x " << object_2_position_base_frame.x << " y " << object_2_position_base_frame.y << " orientation " << object_2_orientation);
  ROS_INFO_STREAM("Object 3 position base frame: x " << object_3_position_base_frame.x << " y " << object_3_position_base_frame.y << " orientation " << object_3_orientation);
  ROS_INFO_STREAM("Object 4 position base frame: x " << object_4_position_base_frame.x << " y " << object_4_position_base_frame.y << " orientation " << object_4_orientation);
}

// service call response
bool get_box_and_target_position(opencv_services::box_and_target_position::Request  &req,
    opencv_services::box_and_target_position::Response &res) {
      res.x1 = object_1_position_base_frame.x;
      res.y1 = object_1_position_base_frame.y;
      res.a1 = object_1_orientation;
      res.c1 = object_1_class;
    
      res.x2 = object_2_position_base_frame.x;
      res.y2 = object_2_position_base_frame.y;
      res.a2 = object_2_orientation;
      res.c2 = object_2_class;

      res.x3 = object_3_position_base_frame.x;
      res.y3 = object_3_position_base_frame.y;
      res.a3 = object_3_orientation;
      res.c3 = object_3_class;

      res.x4 = object_4_position_base_frame.x;
      res.y4 = object_4_position_base_frame.y;
      res.a4 = object_4_orientation;
      res.c4 = object_4_class;

      return true;
    }

void get_centroid(const custom_msg::custom msg){
  // ROS_INFO_STREAM("Object 1 msg: x " << msg.a1 << " y " << unsigned(msg.c1));

  object_1_centroid.x = msg.x1;
  object_1_centroid.y = msg.y1;
  object_1_orientation = msg.a1;
  object_1_class = msg.c1;

  object_2_centroid.x = msg.x2;
  object_2_centroid.y = msg.y2;
  object_2_orientation = msg.a2;
  object_2_class = msg.c2;

  object_3_centroid.x = msg.x3;
  object_3_centroid.y = msg.y3;
  object_3_orientation = msg.a3;
  object_3_class = msg.c3;

  object_4_centroid.x = msg.x4;
  object_4_centroid.y = msg.y4;
  object_4_orientation = msg.a4;
  object_4_class = msg.c4;
  
}

 // Main function
int main(int argc, char **argv)
{
  // The name of the node
  ros::init(argc, argv, "opencv_services");
   
  // Default handler for nodes in ROS
  ros::NodeHandle nh("");


  ros::Subscriber sub = nh.subscribe("/detection_topic", 1000, get_centroid);









    // Used to publish and subscribe to images
  // image_transport::ImageTransport it(nh);

    // Subscribe to the /camera raw image topic
  // image_transport::Subscriber image_sub = it.subscribe(IMAGE_TOPIC, 1, image_cb);

    // Subscribe to the /camera PointCloud2 topic
  ros::Subscriber point_cloud_sub = nh.subscribe(POINT_CLOUD2_TOPIC, 1, point_cloud_cb);
  
  tf2_ros::TransformListener listener(tf_buffer);
   
  ros::ServiceServer service = nh.advertiseService("box_and_target_position",  get_box_and_target_position);
   
  // Make sure we keep reading new video frames by calling the imageCallback function
  ros::spin();
   
  // Close down OpenCV
  cv::destroyWindow("view");
}

 

