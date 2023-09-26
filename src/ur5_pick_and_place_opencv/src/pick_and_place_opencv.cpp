#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include "opencv_services/box_and_target_position.h"


int main(int argc, char **argv)
{
  ros::init(argc, argv, "move_group_interface_tutorial");
  ros::NodeHandle n;

  // ROS spinning must be running for the MoveGroupInterface to get information
  // about the robot's state. One way to do this is to start an AsyncSpinner
  // beforehand.
  ros::AsyncSpinner spinner(1);
  spinner.start();

  // MoveIt operates on sets of joints called "planning groups" and stores them in an object called
  // the `JointModelGroup`. Throughout MoveIt the terms "planning group" and "joint model group"
  // are used interchangably.
  static const std::string PLANNING_GROUP_ARM = "ur5_arm";
  static const std::string PLANNING_GROUP_GRIPPER = "gripper";


  // Define boxes positions
  geometry_msgs::Point green_box_position;
  green_box_position.x=0.3;
  green_box_position.y=-0.5;
  green_box_position.z=0.4;
  geometry_msgs::Point blue_box_position;
  blue_box_position.x=-0.3;
  blue_box_position.y=-0.5;
  blue_box_position.z=0.4;

  // The :planning_interface:`MoveGroupInterface` class can be easily
  // setup using just the name of the planning group you would like to control and plan for.
  moveit::planning_interface::MoveGroupInterface move_group_interface_arm(PLANNING_GROUP_ARM);
  moveit::planning_interface::MoveGroupInterface move_group_interface_gripper(PLANNING_GROUP_GRIPPER);







 












  // We can get a list of all the groups in the robot:
  ROS_INFO_NAMED("tutorial", "Available Planning Groups:");
  std::copy(move_group_interface_arm.getJointModelGroupNames().begin(),
            move_group_interface_arm.getJointModelGroupNames().end(), std::ostream_iterator<std::string>(std::cout, ", "));

  // Get object positions, class and orientation from the opencv node
  ros::ServiceClient box_and_target_position_srv_client = n.serviceClient<opencv_services::box_and_target_position>("box_and_target_position");

  opencv_services::box_and_target_position srv;

  if(box_and_target_position_srv_client.call(srv)) {
    ROS_INFO_STREAM("Object 1 position camera frame: x " << srv.response.x1 << " y " << srv.response.y1 << " orientation " << srv.response.a1 << " class " << srv.response.c1);
    ROS_INFO_STREAM("Object 2 position camera frame: x " << srv.response.x2 << " y " << srv.response.y2 << " orientation " << srv.response.a2 << " class " << srv.response.c2);
    ROS_INFO_STREAM("Object 3 position camera frame: x " << srv.response.x3 << " y " << srv.response.y3 << " orientation " << srv.response.a3 << " class " << srv.response.c3);
    ROS_INFO_STREAM("Object 4 position camera frame: x " << srv.response.x4 << " y " << srv.response.y4 << " orientation " << srv.response.a4 << " class " << srv.response.c4);
  } else {
    ROS_INFO_STREAM("Failed to call box and target position service");
  }




  float objects_positions[4][4]={
    {srv.response.x1, srv.response.y1, srv.response.a1, srv.response.c1},
    {srv.response.x2, srv.response.y2, srv.response.a2, srv.response.c2},
    {srv.response.x3, srv.response.y3, srv.response.a3, srv.response.c3},
    {srv.response.x4, srv.response.y4, srv.response.a4, srv.response.c4}
  }; 

  for (int i = 0; i < 4; i++)
  {
    if(int(objects_positions[i][3]) == 255){
      continue;
    }
    geometry_msgs::Point target_box_position;
    std::string object;
    switch (int(objects_positions[i][3]))
    {
    case 1:
      object="apple";
      target_box_position=green_box_position;
      break;
    case 2:
      object="banana";
      target_box_position=green_box_position;
      break;
    case 3:
      object="can";
      target_box_position=blue_box_position;
      break;
    case 4:
      object="screwdriver";
      target_box_position=blue_box_position;
      break;
    default:
      break;
    }

    ROS_INFO_STREAM("Find " << object << " in position (" << objects_positions[i][0] << " , " << objects_positions[i][1] << ")");

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    robot_model_loader::RobotModelLoaderPtr robot_model_loader(new robot_model_loader::RobotModelLoader("robot_description"));
    planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor(new planning_scene_monitor::PlanningSceneMonitor(robot_model_loader));


    // Add the object to be grasped (the suqare box) to the planning scene
    moveit_msgs::CollisionObject collision_object;
    collision_object.header.frame_id = move_group_interface_arm.getPlanningFrame();

    collision_object.id = "blue_box";

    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 0.02;
    primitive.dimensions[1] = 0.02;
    primitive.dimensions[2] = 0.037;

    geometry_msgs::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = objects_positions[i][0];
    box_pose.position.y = objects_positions[i][1];
    box_pose.position.z = 1.045 - 1.21;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    std::vector<moveit_msgs::CollisionObject> collision_objects;
    collision_objects.push_back(collision_object);

    planning_scene_interface.applyCollisionObjects(collision_objects);

    ROS_INFO_NAMED("tutorial", "Add an object into the world");

    ros::Duration(0.1).sleep();

    // Allow collisions between the gripper and the box to be able to grasp it
    planning_scene_monitor::LockedPlanningSceneRW ls(planning_scene_monitor);
    collision_detection::AllowedCollisionMatrix& acm = ls->getAllowedCollisionMatrixNonConst();
    acm.setEntry("blue_box", "robotiq_85_left_finger_tip_link", true);
    acm.setEntry("blue_box", "robotiq_85_right_finger_tip_link", true);
    acm.setEntry("blue_box", "robotiq_85_right_inner_knuckle_link", true);
    acm.setEntry("blue_box","robotiq_85_left_inner_knuckle_link", true);
    acm.setEntry("blue_box","robotiq_85_base_link", true);
    acm.setEntry("blue_box", "robotiq_85_left_finger_link", true);
    acm.setEntry("blue_box", "robotiq_85_right_finger_link", true);
    acm.setEntry("blue_box", "robotiq_85_left_knuckle_link", true);
    acm.setEntry("blue_box", "robotiq_85_right_knuckle_link", true);
    
    std::cout << "\nAllowedCollisionMatrix:\n";
    acm.print(std::cout);
    moveit_msgs::PlanningScene diff_scene;
    ls->getPlanningSceneDiffMsg(diff_scene);

    planning_scene_interface.applyPlanningScene(diff_scene); 

    ros::Duration(0.1).sleep();












  

    moveit::planning_interface::MoveGroupInterface::Plan my_plan_arm;

    // Move to home position
    move_group_interface_arm.setJointValueTarget(move_group_interface_arm.getNamedTargetValues("home"));

    bool success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Move to home position %s", success ? "" : "FAILED");

    move_group_interface_arm.move();

    // Rotate the gripper
    move_group_interface_arm.setJointValueTarget("wrist_3_joint", (-1*objects_positions[i][2])+1.5708);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Rotate the gripper %s", success ? "" : "FAILED");

    move_group_interface_arm.move();



    // Place the gripper above the object
    geometry_msgs::PoseStamped current_pose;
    current_pose = move_group_interface_arm.getCurrentPose("ee_link");

    geometry_msgs::Pose target_pose1;




    target_pose1.orientation = current_pose.pose.orientation;
    target_pose1.position.x = 0.3;
    target_pose1.position.y = 0.3;
    target_pose1.position.z = current_pose.pose.position.z;
    move_group_interface_arm.setPoseTarget(target_pose1);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Place the Gripper above the object %s", success ? "" : "FAILED");

    ROS_INFO_STREAM(" x " << current_pose.pose.orientation.x << " y " << current_pose.pose.orientation.y << " z " << current_pose.pose.orientation.z << " w " << current_pose.pose.orientation.w);

    move_group_interface_arm.move();

    // target_pose1.orientation = current_pose.pose.orientation;
    // target_pose1.position.x = 0.15;
    // target_pose1.position.y = 0.4;
    // target_pose1.position.z = current_pose.pose.position.z;
    // move_group_interface_arm.setPoseTarget(target_pose1);

    // success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    // ROS_INFO_NAMED("tutorial", "Place the Gripper above the object %s", success ? "" : "FAILED");

    // move_group_interface_arm.move();

    target_pose1.orientation = current_pose.pose.orientation;
    target_pose1.position.x = 0;
    target_pose1.position.y = 0.5;
    target_pose1.position.z = current_pose.pose.position.z;
    move_group_interface_arm.setPoseTarget(target_pose1);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Place the Gripper above the object %s", success ? "" : "FAILED");

    move_group_interface_arm.move();




    


    


    target_pose1.orientation = current_pose.pose.orientation;
    target_pose1.position.x = objects_positions[i][0];
    target_pose1.position.y = objects_positions[i][1];
    target_pose1.position.z = 0.2;
    move_group_interface_arm.setPoseTarget(target_pose1);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    // ROS_INFO_NAMED("tutorial", "Place the Gripper above the object %s", success ? "" : "FAILED");

    move_group_interface_arm.move();

    moveit::planning_interface::MoveGroupInterface::Plan my_plan_gripper;

    // Open the gripper
    move_group_interface_gripper.setJointValueTarget(move_group_interface_gripper.getNamedTargetValues("open"));

    success = (move_group_interface_gripper.plan(my_plan_gripper) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Open the gripper %s", success ? "" : "FAILED");

    move_group_interface_gripper.move();

    // Move the gripper close to the object

    if(int(objects_positions[i][3]) == 1){
      target_pose1.position.z = target_pose1.position.z - 0.2;
    }else{
      target_pose1.position.z = target_pose1.position.z - 0.22;
    }

    move_group_interface_arm.setPoseTarget(target_pose1);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Move the gripper close to the object %s", success ? "" : "FAILED");

    move_group_interface_arm.move();

    // Close the  gripper

    move_group_interface_gripper.setJointValueTarget(move_group_interface_gripper.getNamedTargetValues("closed_"+object));

    success = (move_group_interface_gripper.plan(my_plan_gripper) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Close the  gripper %s", success ? "" : "FAILED");

    move_group_interface_gripper.move();




    // Attach the object to the gripper after it was grasped
    moveit_msgs::AttachedCollisionObject aco;
    aco.object.id = collision_object.id;
    aco.link_name = "robotiq_85_right_finger_tip_link";
    aco.touch_links.push_back("robotiq_85_left_finger_tip_link");
    aco.object.operation = moveit_msgs::CollisionObject::ADD;
    planning_scene_interface.applyAttachedCollisionObject(aco);





    // Move to home position
    // move_group_interface_arm.setJointValueTarget(move_group_interface_arm.getNamedTargetValues("home"));

    // success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    // ROS_INFO_NAMED("tutorial", "Move to home position %s", success ? "" : "FAILED");

    // move_group_interface_arm.move();

    // current_pose = move_group_interface_arm.getCurrentPose("ee_link");

    // target_pose1.orientation = current_pose.pose.orientation;
    // target_pose1.position.x = 0.5;
    // target_pose1.position.y = 0;
    // target_pose1.position.z = 0.6;
    // move_group_interface_arm.setPoseTarget(target_pose1);

    // success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    // ROS_INFO_NAMED("tutorial", "Place the Gripper above the object %s", success ? "" : "FAILED");

    // move_group_interface_arm.move();



    // Rotate the gripper
    move_group_interface_arm.setJointValueTarget("wrist_3_joint", 1.5708);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Rotate the gripper %s", success ? "" : "FAILED");

    move_group_interface_arm.move();


    current_pose = move_group_interface_arm.getCurrentPose("ee_link");

    target_pose1.orientation = current_pose.pose.orientation;
    target_pose1.position.x = 0;
    target_pose1.position.y = -0.5;
    target_pose1.position.z = current_pose.pose.position.z;
    move_group_interface_arm.setPoseTarget(target_pose1);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Place the Gripper above the object %s", success ? "" : "FAILED");

    move_group_interface_arm.move();


    // Move the gripper above the box


    current_pose = move_group_interface_arm.getCurrentPose("ee_link");

    target_pose1.orientation = current_pose.pose.orientation;


    target_pose1.position=target_box_position;
   
    move_group_interface_arm.setPoseTarget(target_pose1);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Move the gripper above the box %s", success ? "" : "FAILED");

    move_group_interface_arm.move();

    // Lower the gripper inside the box
    target_pose1.position.z = target_pose1.position.z - 0.14;
    move_group_interface_arm.setPoseTarget(target_pose1);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Lower the gripper inside the box %s", success ? "" : "FAILED");

    move_group_interface_arm.move();

    // Open the gripper
    move_group_interface_gripper.setJointValueTarget(move_group_interface_gripper.getNamedTargetValues("open"));

    success = (move_group_interface_gripper.plan(my_plan_gripper) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Open the gripper %s", success ? "" : "FAILED");

    move_group_interface_gripper.move();


    aco.object.operation = moveit_msgs::CollisionObject::REMOVE;
    planning_scene_interface.applyAttachedCollisionObject(aco);



    // Upper the gripper outside the box
    target_pose1.position.z = target_pose1.position.z + 0.14;
    move_group_interface_arm.setPoseTarget(target_pose1);

    success = (move_group_interface_arm.plan(my_plan_arm) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    ROS_INFO_NAMED("tutorial", "Upper the gripper outside the box %s", success ? "" : "FAILED");

    move_group_interface_arm.move();

    ROS_INFO_NAMED("tutorial", "Remove the object from the world");
    std::vector<std::string> object_ids;
    object_ids.push_back(collision_object.id);
    planning_scene_interface.removeCollisionObjects(object_ids);
  }

  ros::shutdown();
  return 0;
}
