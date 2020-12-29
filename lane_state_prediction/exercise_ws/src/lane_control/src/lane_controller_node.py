#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, Segment, SegmentList
from geometry_msgs.msg import Point
from sensor_msgs.msg import Joy
import os
from lane_controller.controller import PurePursuitLaneController
import pickle
from sklearn.gaussian_process import GaussianProcessClassifier
import json


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

        self.max_lookahead_GP = rospy.get_param('~max_lookahead_GP', 0.4)
        self.min_lookahead_GP = rospy.get_param('~min_lookahead_GP', 0.1)
        self.train_filename_GP = rospy.get_param('~train_filename_GP', None)
        self.GP_model_file = rospy.get_param('~GP_model_file', None)
        self.predict_lane_state = rospy.get_param('~predict_lane_state', None)

        if self.predict_lane_state:
            dir = "/code/exercise_ws/data/"
            self.model = pickle.load(open(dir+self.GP_model_file, 'rb'))
        
        self.pp_controller = PurePursuitLaneController(self.params)

        # Initialize variables
        self.target_point = Point()
        self.last_s = 0.0
        self.direction = "straight"
        self.training_data = []

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
        self.sub_line_seg_reading = rospy.Subscriber(f"/{os.environ['VEHICLE_NAME']}/ground_projection_node/lineseglist_out",
                                                 SegmentList,
                                                 self.cbLineSegList,
                                                 queue_size=1)
        self.sub_joy_ = rospy.Subscriber("joy", Joy, self.cbJoy, queue_size=1)
        self.log("Initialized!")

    def cbJoy(self, joy_msg):
        self.joy = joy_msg
        # S key pressed
        if joy_msg.buttons[6] == 1:
            print("S pressed")
            self.save_toggl = False
        elif joy_msg.buttons[3] == 1:
            print("E pressed")
            self.save_toggl = True

        if self.save_toggl:
            if joy_msg.axes[1] == 1.0:
                print("straight path saved")
                self.saveData("straight")
                self.save_toggl = False
            elif joy_msg.axes[3] == 1.0:
                print("left path saved")
                self.saveData("left")
                self.save_toggl = False
            elif joy_msg.axes[3] == -1.0:
                print("right path saved")
                self.saveData("right")
                self.save_toggl = False
            elif joy_msg.axes[1] == -1.0:
                print("JSON saved")
                self.writeToFile()
                self.save_toggl = False
            
    def saveData(self, direction):
        curr_dict = {"white": self.white_seg_list_data, "yellow": self.yellow_seg_list_data, "direction": direction}
        self.training_data.append(curr_dict)

    def writeToFile(self):
        dir = "/code/exercise_ws/data/"
        with open(dir+self.train_filename, "w") as f:
            json.dump(self.training_data,f)

    def cbLineSegList(self, LineSegList):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        white_seg_list = []
        yellow_seg_list = []
        white_seg_num = 0
        yellow_seg_num = 0
        car_pos = {"x": 0.0, "y": 0.0}      # Segments are in pos relative to car which is at (0, 0)
        white_avg = {"x": 0.0, "y": 0.0}
        white_max = {"x": 0.0, "y": 0.0}
        white_min = {"x": 0.0, "y": 0.0}
        yellow_avg = {"x": 0.0, "y": 0.0}
        yellow_max = {"x": 0.0, "y": 0.0}
        yellow_min = {"x": 0.0, "y": 0.0}

        curr_white_min = 1000.0
        curr_white_max = 0.0
        curr_yellow_min = 1000.0
        curr_yellow_max = 0.0
        for seg in LineSegList.segments:
            # Get midpoint of segment
            mid_x = (seg.points[0].x + seg.points[1].x)/2
            mid_y = (seg.points[0].y + seg.points[1].y)/2
            temp_seg = {"x": mid_x, "y": mid_y}

            # Add simplified segment point to respective colour list
            dist = self.meas_dis(temp_seg, car_pos)
            
            if dist < self.max_lookahead_GP and dist > self.min_lookahead_GP:
                if seg.color == 0:
                    white_seg_list.append(temp_seg)
                    white_avg['x'] = white_avg['x'] + temp_seg['x']
                    white_avg['y'] = white_avg['y'] + temp_seg['y']
                    if dist < curr_white_min:
                        white_min = temp_seg
                        curr_white_min = dist
                    if dist > curr_white_max:
                        white_max = temp_seg
                        curr_white_max = dist
                    white_seg_num += 1
                elif seg.color == 1:
                    yellow_seg_list.append(temp_seg)
                    yellow_avg['x'] = yellow_avg['x'] + temp_seg['x']
                    yellow_avg['y'] = yellow_avg['y'] + temp_seg['y']
                    if dist < curr_yellow_min:
                        yellow_min = temp_seg
                        curr_yellow_min = dist
                    if dist > curr_yellow_max:
                        yellow_max = temp_seg
                        curr_yellow_max = dist
                    yellow_seg_num += 1
        if white_seg_num > 0:
            white_avg['x'] = white_avg['x']/white_seg_num
            white_avg['y'] = white_avg['y']/white_seg_num
        if yellow_seg_num > 0:
            yellow_avg['x'] = yellow_avg['x']/yellow_seg_num
            yellow_avg['y'] = yellow_avg['y']/yellow_seg_num

        self.white_seg_list_data = {"segs": white_seg_list, "avg": white_avg, "min": white_min, "max": white_max}
        self.yellow_seg_list_data = {"segs": yellow_seg_list, "avg": yellow_avg, "min": yellow_min, "max": yellow_max}

        # Also predict lane state based on GP if available
        if self.predict_lane_state:
            x_star = np.array([white_avg['x'], white_avg['y'], white_min['x'], white_min['y'], white_max['x'], white_max['y']\
                , yellow_avg['x'], yellow_avg['y'], yellow_min['x'], yellow_min['y'], yellow_max['x'], yellow_max['y']])
            y_star = self.model.predict([x_star])
            if y_star == 0:
                self.direction = "straight"
            elif y_star == 1:
                self.direction = "left"
            elif y_star == 2:
                self.direction = "right"
            else:
                self.direction = "unknown"
        else: 
            self.direction = "straight"
        
        current_s = rospy.Time.now().to_sec()
        if current_s - self.last_s > self.params['~sample_rate']:
            x_target, y_target = self.pp_controller.update_target_point(self.target_point, self.direction, white_seg_list, yellow_seg_list)
            self.last_s = current_s
        else:
            x_target = self.target_point.x
            y_target = self.target_point.y

        print("")
        print(str(self.direction))
        print(str(x_target))
        print(str(y_target))

        self.target_point.x = x_target
        self.target_point.y = y_target

    def meas_dis(self, seg1, seg2):
        """Calculated the euclidean distance between two single point segments.

            Args:
                seg1 (:obj:`dict`): dictionary containing an x and y coordinate.
                seg2 (:obj:`dict`): dictionary containing an x and y coordinate.
        """
        return ((seg1['x'] - seg2['x'])**2 + (seg1['y'] - seg2['y'])**2)**0.5

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
