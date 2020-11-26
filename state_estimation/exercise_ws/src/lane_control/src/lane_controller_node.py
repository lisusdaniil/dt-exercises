#!/usr/bin/env python3
import numpy as np
import rospy
import os

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, SegmentList

from lane_controller.controller import LaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities, by processing the estimate error in
    lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:
        ~v_bar (:obj:`float`): Nominal velocity in m/s
        ~k_d (:obj:`float`): Proportional term for lateral deviation
        ~k_theta (:obj:`float`): Proportional term for heading deviation
        ~k_Id (:obj:`float`): integral term for lateral deviation
        ~k_Iphi (:obj:`float`): integral term for lateral deviation
        ~d_thres (:obj:`float`): Maximum value for lateral error
        ~theta_thres (:obj:`float`): Maximum value for heading error
        ~d_offset (:obj:`float`): Goal offset from center of the lane
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)
        ~stop_line_slowdown (:obj:`dict`): Start and end distances for slowdown at stop lines

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline, to reduce speed
        ~obstacle_distance_reading (:obj:`stop_line_reading`): Distancefrom obstacle virtual stopline, to reduce speed
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )

        # Add the node parameters to the parameters dictionary
        # TODO: MAKE TO WORK WITH NEW DTROS PARAMETERS
        self.params = dict()
        self.params['~v_bar_st'] = DTParam(
            '~v_bar_st',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=5.0
        )
        self.params['~k_d_st'] = DTParam(
            '~k_d_st',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_der_d_st'] = DTParam(
            '~k_der_d_st',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_der_theta_st'] = DTParam(
            '~k_der_theta_st',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_theta_st'] = DTParam(
            '~k_theta_st',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_Id_st'] = DTParam(
            '~k_Id_st',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_Iphi_st'] = DTParam(
            '~k_Iphi_st',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~v_bar_tr'] = DTParam(
            '~v_bar_tr',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=5.0
        )
        self.params['~k_d_tr'] = DTParam(
            '~k_d_tr',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_theta_tr'] = DTParam(
            '~k_theta_tr',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_der_d_tr'] = DTParam(
            '~k_der_d_tr',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_der_theta_tr'] = DTParam(
            '~k_der_theta_tr',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_Id_tr'] = DTParam(
            '~k_Id_tr',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~k_Iphi_tr'] = DTParam(
            '~k_Iphi_tr',
            param_type=ParamType.FLOAT,
            min_value=-100.0,
            max_value=100.0
        )
        self.params['~omega_max_st'] = rospy.get_param('~omega_max_st', None)
        self.params['~omega_max_tr'] = rospy.get_param('~omega_max_tr', None)
        self.params['~theta_thres'] = rospy.get_param('~theta_thres', None)
        self.params['~d_thres'] = rospy.get_param('~d_thres', None)
        self.params['~d_offset'] = rospy.get_param('~d_offset', None)
        self.params['~integral_bounds'] = rospy.get_param('~integral_bounds', None)
        self.params['~d_resolution'] = rospy.get_param('~d_resolution', None)
        self.params['~phi_resolution'] = rospy.get_param('~phi_resolution', None)
        self.params['~omega_ff'] = rospy.get_param('~omega_ff', None)
        self.params['~verbose'] = rospy.get_param('~verbose', None)
        self.params['~stop_line_slowdown'] = rospy.get_param('~stop_line_slowdown', None)

        # Params for direction detection
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
        self.params['~sample_rate'] = rospy.get_param('~sample_rate', None)
        self.params['~turn_delay'] = rospy.get_param('~turn_delay', None)

        # Need to create controller object before updating parameters, otherwise it will fail
        self.controller = LaneController(self.params)
        # self.updateParameters() # TODO: This needs be replaced by the new DTROS callback when it is implemented

        # Initialize variables
        self.direction = "straight"
        self.dir_list = [self.direction]*10
        self.update_dir = True
        self.fsm_state = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.pose_msg = LanePose()
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.last_s = 0.0
        self.last_s_dir = 0.0
        self.last_s_turn = 0.0
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.obstacle_stop_line_distance = None
        self.obstacle_stop_line_detected = False
        self.at_obstacle_stop_line = False

        self.current_pose_source = 'lane_filter'

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbAllPoses,
                                                 "lane_filter",
                                                 queue_size=1)
        self.sub_intersection_navigation_pose = rospy.Subscriber("~intersection_navigation_pose",
                                                                 LanePose,
                                                                 self.cbAllPoses,
                                                                 "intersection_navigation",
                                                                 queue_size=1)
        #self.sub_wheels_cmd_executed = rospy.Subscriber("~wheels_cmd_executed",
        #                                                WheelsCmdStamped,
        #                                                self.cbWheelsCmdExecuted,
        #                                                queue_size=1)
        self.sub_wheels_cmd_executed = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/wheels_driver_node/wheels_cmd",
                                                        WheelsCmdStamped,
                                                        self.cbWheelsCmdExecuted,
                                                        queue_size=1)
        self.sub_stop_line = rospy.Subscriber("~stop_line_reading",
                                              StopLineReading,
                                              self.cbStopLineReading,
                                              queue_size=1)
        self.sub_obstacle_stop_line = rospy.Subscriber("~obstacle_distance_reading",
                                                        StopLineReading,
                                                        self.cbObstacleStopLineReading,
                                                        queue_size=1)

        self.sub_line_seg_reading = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/ground_projection_node/lineseglist_out",
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
        if current_s - self.last_s_dir > self.params['~sample_rate']:
            direction = self.controller.get_direction(white_seg_list, yellow_seg_list)
            self.last_s_dir = current_s

            if direction != 'unknown':
                self.dir_list.pop(0)
                self.dir_list.append(direction)

            num_st = self.dir_list.count('straight')
            num_ri = self.dir_list.count('right')
            num_le = self.dir_list.count('left')
            num_un = self.dir_list.count('unknown')
            num_tr = max(num_ri, num_le)

            curr_turn_s = rospy.Time.now().to_sec()
            if curr_turn_s - self.last_s_turn > self.params['~turn_delay']:
                self.update_dir = True

            if self.update_dir == True:
                if num_un > 2:
                    self.direction = 'unknown'
                elif num_tr > num_st:
                    self.update_dir = False
                    if num_ri > num_le:
                        self.direction = 'right'
                    else:
                        self.direction = 'left'
                else:
                    self.direction = 'straight'
        #print(self.dir_list)
        #print(self.direction)

    def cbObstacleStopLineReading(self,msg):
        """
        Callback storing the current obstacle distance, if detected.

        Args:
            msg(:obj:`StopLineReading`): Message containing information about the virtual obstacle stopline.
        """
        self.obstacle_stop_line_distance = np.sqrt(msg.stop_line_point.x ** 2 + msg.stop_line_point.y ** 2)
        self.obstacle_stop_line_detected = msg.stop_line_detected
        self.at_stop_line = msg.at_stop_line

    def cbStopLineReading(self, msg):
        """Callback storing current distance to the next stopline, if one is detected.

        Args:
            msg (:obj:`StopLineReading`): Message containing information about the next stop line.
        """
        self.stop_line_distance = np.sqrt(msg.stop_line_point.x ** 2 + msg.stop_line_point.y ** 2)
        self.stop_line_detected = msg.stop_line_detected
        self.at_obstacle_stop_line = msg.at_stop_line

    def cbMode(self, fsm_state_msg):

        self.fsm_state = fsm_state_msg.state  # String of current FSM state

        if self.fsm_state == 'INTERSECTION_CONTROL':
            self.current_pose_source = 'intersection_navigation'
        else:
            self.current_pose_source = 'lane_filter'

        if self.params['~verbose'] == 2:
            self.log("Pose source: %s" % self.current_pose_source)

    def cbAllPoses(self, input_pose_msg, pose_source):
        """Callback receiving pose messages from multiple topics.

        If the source of the message corresponds with the current wanted pose source, it computes a control command.

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
            pose_source (:obj:`String`): Source of the message, specified in the subscriber.
        """

        if pose_source == self.current_pose_source:
            self.pose_msg_dict[pose_source] = input_pose_msg

            self.pose_msg = input_pose_msg

            self.getControlAction(self.pose_msg)

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        """Callback that reports if the requested control action was executed.

        Args:
            msg_wheels_cmd (:obj:`WheelsCmdStamped`): Executed wheel commands
        """
        self.wheels_cmd_executed = msg_wheels_cmd

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)

    def getControlAction(self, pose_msg):
        """Callback that receives a pose message and updates the related control command.

        Using a controller object, computes the control action using the current pose estimate.

        Args:
            pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        current_s = rospy.Time.now().to_sec()
        dt = None
        if self.last_s is not None:
            dt = (current_s - self.last_s)

        if self.at_stop_line or self.at_obstacle_stop_line:
            v = 0
            omega = 0
        else:
               
            # Compute errors
            d_err = pose_msg.d - self.params['~d_offset']
            phi_err = pose_msg.phi

            # We cap the error if it grows too large
            if np.abs(d_err) > self.params['~d_thres']:
                self.log("d_err too large, thresholding it!", 'error')
                d_err = np.sign(d_err) * self.params['~d_thres']


            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
            if self.obstacle_stop_line_detected:
                v, omega = self.controller.compute_control_action(d_err, phi_err, dt, wheels_cmd_exec, self.obstacle_stop_line_distance, self.direction)
                #TODO: This is a temporarily fix to avoid vehicle image detection latency caused unable to stop in time.
                v = v*0.25
                omega = omega*0.25

            else:
                v, omega = self.controller.compute_control_action(d_err, phi_err, dt, wheels_cmd_exec, self.stop_line_distance, self.direction)

            # For feedforward action (i.e. during intersection navigation)
            omega += self.params['~omega_ff']

        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = pose_msg.header

        # Add commands to car message
        #print("v: " + str(v))
        #print("omega: " + str(omega))
        car_control_msg.v = v
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)
        self.last_s = current_s

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
