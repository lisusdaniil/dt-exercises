#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, Segment, SegmentList
from geometry_msgs.msg import Point

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.params['~look_ahead_dist'] = rospy.get_param('~look_ahead_dist', None)
        self.params['~max_lookahead'] = rospy.get_param('~max_lookahead', None)
        self.params['~min_lookahead'] = rospy.get_param('~min_lookahead', None) 
        self.params['~seg_collapse_dist'] = rospy.get_param('~seg_collapse_dist', None)
        self.params['~seg_group_dist_x'] = rospy.get_param('~seg_group_dist_x', None)
        self.params['~seg_group_dist_y'] = rospy.get_param('~seg_group_dist_y', None)
        self.params['~correction_from_edge'] = rospy.get_param('~correction_from_edge', None)
        self.params['~horizontal_distribution'] = rospy.get_param('~horizontal_distribution', None)
        self.params['~max_yellow_white_dist'] = rospy.get_param('~max_yellow_white_dist', None)
        self.params['~pos_shift_bc_curve'] = rospy.get_param('~pos_shift_bc_curve', None)
        self.params['~K'] = rospy.get_param('~K', None)
        self.params['~v_nom'] = rospy.get_param('~v_nom', None)
        self.params['~v_min'] = rospy.get_param('~v_min', None)
        self.params['~alpha_K'] = rospy.get_param('~alpha_K', None)
        self.params['~sample_rate'] = rospy.get_param('~sample_rate', None)
        
        self.pp_controller = PurePursuitLaneController(self.params)

        # Initialize variables
        self.target_point = Point()
        self.last_s = 0.2
        self.direction = "unknown"

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)
        self.sub_line_seg_reading = rospy.Subscriber("/agent/ground_projection_node/lineseglist_out",
                                                 SegmentList,
                                                 self.cbLineSegList,
                                                 queue_size=1)

        self.log("Initialized!")

    def cbLineSegList(self, LineSegList):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        white_seg_list = []
        yellow_seg_list = []
        seg_num = 0
        for seg in LineSegList.segments:
            # Get midpoint of segment
            mid_x = (seg.points[0].x + seg.points[1].x)/2
            mid_y = (seg.points[0].y + seg.points[1].y)/2
            temp_dict = {"x": mid_x, "y": mid_y}

            # Add simplified segment point to respective colour list
            if seg.color == 0:
                white_seg_list.append(temp_dict)
                seg_num += 1
            elif seg.color == 1:
                yellow_seg_list.append(temp_dict)

        current_s = rospy.Time.now().to_sec()
        if current_s - self.last_s > self.params['~sample_rate']:
            x_target, y_target, direction = self.pp_controller.update_target_point(self.target_point, white_seg_list, yellow_seg_list)
            self.last_s = current_s
        else:
            x_target = self.target_point.x
            y_target = self.target_point.y
            direction = self.direction

        print(direction)
        print(x_target)
        print(y_target)

        self.target_point.x = x_target
        self.target_point.y = y_target
        self.direction = direction


    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # TODO This needs to get changed
        v, omega = self.pp_controller.compute_control_action(self.target_point, self.params['~K'])
        car_control_msg.v = v
        print(v)
        print("\n\n\n")
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)


    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
