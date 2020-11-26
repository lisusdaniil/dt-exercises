import numpy as np


class LaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    This implementation is a simple PI(D) controller.

    Args:
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

    """
    def __init__(self, parameters):
        self.parameters = parameters
        self.d_I = 0.0
        self.phi_I = 0.0
        self.prev_d_err = 0.0
        self.prev_phi_err = 0.0

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters

    def compute_control_action(self, d_err, phi_err, dt, wheels_cmd_exec, stop_line_distance, direction):
        """Main function, computes the control action given the current error signals.

        Given an estimate of the error, computes a control action (tuple of linear and angular velocity). This is done
        via a basic PI(D) controller with anti-reset windup logic.

        Args:
            d_err (:obj:`float`): error in meters in the lateral direction
            phi_err (:obj:`float`): error in radians in the heading direction
            dt (:obj:`float`): time since last command update
            wheels_cmd_exec (:obj:`bool`): confirmation that the wheel commands have been executed (to avoid
                                           integration while the robot does not move)
            stop_line_distance (:obj:`float`):  distance of the stop line, None if not detected.
        Returns:
            v (:obj:`float`): requested linear velocity in meters/second
            omega (:obj:`float`): requested angular velocity in radians/second
        """
        k_d, k_theta, k_der_d, k_der_theta, k_Id, k_Iphi = self.schedule_values(direction)

        d_der = 0.0
        phi_der = 0.0
        if dt is not None:
            self.integrate_errors(d_err, phi_err, dt)
            d_der = (d_err - self.prev_d_err)/dt
            phi_der = (phi_err - self.prev_phi_err)/dt
            
            # Clamp derivatives so they dont blow up for very short timestamps
            if abs(d_der) < 0.001:
                d_der = 0.0

            if abs(phi_der) < 0.001:
                phi_der = 0.0

        self.d_I = self.adjust_integral(d_err, self.d_I, self.parameters['~integral_bounds']['d'],
                                        self.parameters['~d_resolution'])
        self.phi_I = self.adjust_integral(phi_err, self.phi_I, self.parameters['~integral_bounds']['phi'],
                                          self.parameters['~phi_resolution'])

        self.reset_if_needed(d_err, phi_err, wheels_cmd_exec, direction)

        # Scale the parameters linear such that their real value is at 0.22m/s
        omega = k_d * d_err + k_theta * phi_err\
                + k_der_d * d_der + k_der_theta * phi_der \
                + k_Id * self.d_I + k_Iphi * self.phi_I

        #print(k_d * d_err)
        #print(k_der_d * d_der)
        #print(k_Id * self.d_I)
        #print("theta")
        #print(k_theta * phi_err)
        #print(k_der_theta * phi_der)
        #print(k_Iphi * self.phi_I)
        #print("Omega raw: " + str(omega))

        self.prev_d_err = d_err
        self.prev_phi_err = phi_err

        if abs(omega) < 0.001:
            omega = 0.0
        else:
            sign = omega/abs(omega)
            if direction == 'left' or direction == 'right':
                omega = min(abs(omega), self.parameters['~omega_max_tr'])
            else:
                omega = min(abs(omega), self.parameters['~omega_max_st'])
            omega = sign*omega

        v = self.compute_velocity(stop_line_distance, direction)

        return v, omega

    def schedule_values(self, direction):
        # Set default as driving straight parameters
        k_d = self.parameters['~k_d_st'].value
        k_theta = self.parameters['~k_theta_st'].value
        k_der_d = self.parameters['~k_der_d_st'].value
        k_der_theta = self.parameters['~k_der_theta_st'].value
        k_Id = self.parameters['~k_Id_st'].value
        k_Iphi = self.parameters['~k_Iphi_st'].value

        if direction == 'left' or direction == 'right':
            k_d = self.parameters['~k_d_tr'].value
            k_theta = self.parameters['~k_theta_tr'].value
            k_der_d = self.parameters['~k_der_d_tr'].value
            k_der_theta = self.parameters['~k_der_theta_tr'].value
            k_Id = self.parameters['~k_Id_tr'].value
            k_Iphi = self.parameters['~k_Iphi_tr'].value
        elif direction == 'unknown':
            k_d = self.parameters['~k_d_un'].value
            k_theta = self.parameters['~k_theta_un'].value
            k_der_d = self.parameters['~k_der_d_un'].value
            k_der_theta = self.parameters['~k_der_theta_un'].value
            k_Id = self.parameters['~k_Id_un'].value
            k_Iphi = self.parameters['~k_Iphi_un'].value

        return k_d, k_theta, k_der_d, k_der_theta, k_Id, k_Iphi

    def compute_velocity(self, stop_line_distance, direction):
        """Linearly decrease velocity if approaching a stop line.

        If a stop line is detected, the velocity is linearly decreased to achieve a better stopping position,
        otherwise the nominal velocity is returned.

        Args:
            stop_line_distance (:obj:`float`): distance of the stop line, None if not detected.
        """
        if stop_line_distance is None:
            if direction == 'left' or direction == 'right':
                return self.parameters['~v_bar_tr'].value
            else:
                return self.parameters['~v_bar_st'].value
        else:

            d1, d2 = self.parameters['~stop_line_slowdown']['start'], self.parameters['~stop_line_slowdown']['end']
            # d1 -> v_bar, d2 -> v_bar/2
            c = (0.5 * (d1 - stop_line_distance) + (stop_line_distance - d2)) / (d1 - d2)
            v_new = self.parameters['~v_bar_st'].value * c
            v = np.max([self.parameters['~v_bar_st'].value / 2.0, np.min([self.parameters['~v_bar_st'].value, v_new])])
            return v

    def integrate_errors(self, d_err, phi_err, dt):
        """Integrates error signals in lateral and heading direction.
        Args:
            d_err (:obj:`float`): error in meters in the lateral direction
            phi_err (:obj:`float`): error in radians in the heading direction
            dt (:obj:`float`): time delay in seconds
        """
        self.d_I += d_err * dt
        self.phi_I += phi_err * dt

    def reset_if_needed(self, d_err, phi_err, wheels_cmd_exec, direction):
        """Resets the integral error if needed.

        Resets the integral errors in `d` and `phi` if either the error sign changes, or if the robot is completely
        stopped (i.e. intersections).

        Args:
            d_err (:obj:`float`): error in meters in the lateral direction
            phi_err (:obj:`float`): error in radians in the heading direction
            wheels_cmd_exec (:obj:`bool`): confirmation that the wheel commands have been executed (to avoid
                                           integration while the robot does not move)
        """
        if np.sign(d_err) != np.sign(self.prev_d_err):
            self.d_I = 0
        if np.sign(phi_err) != np.sign(self.prev_phi_err):
            self.phi_I = 0
        if direction == 'left' or direction == 'right':
            self.d_I = 0
            self.phi_I = 0
        if wheels_cmd_exec[0] == 0 and wheels_cmd_exec[1] == 0:
            self.d_I = 0
            self.phi_I = 0

    @staticmethod
    def adjust_integral(error, integral, bounds, resolution):
        """Bounds the integral error to avoid windup.

        Adjusts the integral error to remain in defined bounds, and cancels it if the error is smaller than the
        resolution of the error estimation.

        Args:
            error (:obj:`float`): current error value
            integral (:obj:`float`): current integral value
            bounds (:obj:`dict`): contains minimum and maximum value for the integral
            resolution (:obj:`float`): resolution of the error estimate

        Returns:
            integral (:obj:`float`): adjusted integral value
        """
        if integral > bounds['top']:
            integral = bounds['top']
        elif integral < bounds['bot']:
            integral = bounds['bot']
        elif abs(error) < resolution:
            integral = 0
        return integral

    def get_direction(self, white_seg_full, yellow_seg_full):
        """

            Args:
                
        """
        if len(white_seg_full) == 0 and len(yellow_seg_full) == 0:
            #print("seg_full empty")
            return "unknown"

        white_seg_filt = self.filter_segment_list(white_seg_full)
        yellow_seg_filt = self.filter_segment_list(yellow_seg_full)

        if len(white_seg_filt) == 0 and len(yellow_seg_filt) == 0:
            #print("seg_filt empty")
            return "unknown"

        # Sort to maybe help grouping, not really sure if this has impact but why not
        white_seg_filt.sort(key=self.sort_by_y)
        # Group together segments to hopefully form lanes
        white_seg_groups = self.group_segs_together(white_seg_filt)
        yellow_seg_groups = self.group_segs_together(yellow_seg_filt)

        if len(white_seg_groups) == 0 and len(yellow_seg_groups)==0:
            return "unknown"

        # Sort in ascending vertical direction
        for group in white_seg_groups:
            group.sort(key=self.sort_by_x)
        for group in yellow_seg_groups:
            group.sort(key=self.sort_by_x)

        # Try to predict direction of lane
        direction = self.get_lane_groups(white_seg_groups, yellow_seg_groups)
        return direction

    def sort_by_y(self, e):
        return e['y']

    def sort_by_x(self, e):
        return e['x']

    def group_segs_together(self, filtered_segments):
        seg_group_dist_x = self.parameters['~seg_group_dist_x'] 
        seg_group_dist_y = self.parameters['~seg_group_dist_y'] 

        list_of_seg_groups = []
        for curr_seg in filtered_segments:
            # Find closest segment that is not already part of a group we are in
            new_seg_found = False
            excluded_segs = [curr_seg]
            while len(excluded_segs) != len(filtered_segments) and not new_seg_found:
                closest_seg, dist, dist_x, dist_y = self.find_closest_seg(curr_seg, excluded_segs, filtered_segments)
                excluded_segs.append(closest_seg)

                # Check if this closest seg is already in the same group as us
                already_grouped = False
                for group in list_of_seg_groups:
                    if curr_seg in group and closest_seg in group:
                        already_grouped = True
                
                if not already_grouped:
                    new_seg_found = True
            
            # Check if while loop exited bc every segment is in the same group, at this point we're done
            if len(excluded_segs) == len(filtered_segments):
                break

            # Get our current group (if we have one)
            new_group = [curr_seg]
            for group in list_of_seg_groups:
                if curr_seg in group:
                    new_group = group
                    list_of_seg_groups.remove(group)

            # Check if closest segment forms a group with current segment
            if dist_x < seg_group_dist_x and dist_y < seg_group_dist_y:
                # Check if closest segment not in our group belongs to a different group and merge it with our group if yes
                # Otherwise just append it to our group
                new_group.append(closest_seg)
                for group in list_of_seg_groups:
                    if closest_seg in group:
                        new_group.remove(closest_seg)
                        new_group.extend(group)
                        list_of_seg_groups.remove(group)
            
            # Add the newest group to the list of groups
            list_of_seg_groups.append(new_group)

        # Get rid of groups of single segments as these are untrustworthy
        for group in list_of_seg_groups:
            if len(group) < 2:
                list_of_seg_groups.remove(group)

        return list_of_seg_groups

    def find_closest_seg(self, target_seg, excluded_segs, seg_list):
        closest_seg = {}
        min_dist = 10000.0      # Some silly large number
        for seg in seg_list:
            if seg not in excluded_segs:
                dist = self.meas_dis(target_seg, seg)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_x = abs(seg['x'] - target_seg['x'])
                    min_dist_y = abs(seg['y'] - target_seg['y'])
                    closest_seg = seg
        return closest_seg, min_dist, min_dist_x, min_dist_y

    def meas_dis(self, seg1, seg2):
        """Calculated the euclidean distance between two single point segments.

            Args:
                seg1 (:obj:`dict`): dictionary containing an x and y coordinate.
                seg2 (:obj:`dict`): dictionary containing an x and y coordinate.
        """
        return ((seg1['x'] - seg2['x'])**2 + (seg1['y'] - seg2['y'])**2)**0.5

    def filter_segment_list(self, full_seg_list):
        """Filters a single point segment lists by eliminating points too far away and by grouping points that are close to each other.

            Args:
                full_seg_list (:obj:`dict`): dictionary containing single point segments.
        """

        car_pos = {"x": 0.0, "y": 0.0}      # Segments are in pos relative to car which is at (0, 0)
        # Group segments if they are close to each other
        seg_collapse_dist = self.parameters['~seg_collapse_dist']
        max_lookahead = self.parameters['~max_lookahead']
        min_lookahead = self.parameters['~min_lookahead']

        # Group white segments
        filtered_seg_list = full_seg_list.copy()
        ii = 0
        while ii < len(filtered_seg_list):
            curr_seg = filtered_seg_list[ii]
            
            # Delete segment if it is too far away
            if self.meas_dis(curr_seg, car_pos) > max_lookahead:
                filtered_seg_list.pop(ii)
                continue
            # Delete segment if it is too close
            if self.meas_dis(curr_seg, car_pos) < min_lookahead:
                filtered_seg_list.pop(ii)
                continue
            
            for jj in range(ii+1, len(filtered_seg_list)):
                test_seg = filtered_seg_list[jj]

                # If a close point is found, group ii and jj point and save it in ii slot, delete jj point
                if self.meas_dis(curr_seg, test_seg) <= seg_collapse_dist:
                    temp_x = (curr_seg['x'] + test_seg['x'])/2
                    temp_y = (curr_seg['y'] + test_seg['y'])/2
                    filtered_seg_list[ii] = {"x": temp_x, "y": temp_y}
                    filtered_seg_list.pop(jj)
                    break   # Want to redo the loop with the new ii point
                
                # If no close points found, move onto next ii
                if jj == len(filtered_seg_list)-1:
                    ii += 1
            
            # Needed to exit loop while still completing above checks for the last value
            if ii == len(filtered_seg_list)-1:
                ii += 1

        return filtered_seg_list

    def get_lane_groups(self, white_seg_groups, yellow_seg_groups):
        horizontal_distribution = self.parameters['~horizontal_distribution']
        pos_shift_bc_curve = self.parameters['~pos_shift_bc_curve']

        max_len_w = 0
        sec_max_len_w = 0
        min_right_avg_w = 10000
        max_top_avg_w = -1000
        max_bot_avg_w = 10000
        right_group_w = []
        top_group_w = []
        bot_group_w = []

        min_left_avg_y = 10000
        max_top_avg_y = -1000
        max_bot_avg_y = 10000
        left_group_y = []
        top_group_y = []
        bot_group_y = []

        # Process white groups
        for group in white_seg_groups:
            if len(group) > max_len_w:
                max_len_w = len(group)
                max_group = group
            elif len(group) > sec_max_len_w:
                sec_max_len_w = len(group)
            
            side_avg = 0
            height_avg = 0
            for seg in group:
                side_avg = side_avg + seg['y']
                height_avg = height_avg + seg['x']
            side_avg = side_avg/len(group)
            height_avg = height_avg/len(group)

            # We want closest white line to our right
            if (side_avg < pos_shift_bc_curve and side_avg < min_right_avg_w) or len(right_group_w)==0:
                min_right_avg_w = side_avg
                right_group_w = group
            if height_avg > max_top_avg_w or len(top_group_w)==0:
                max_top_avg_w = height_avg
                top_group_w = group
                top_group_side_avg_w = side_avg
            if height_avg < max_bot_avg_w or len(bot_group_w)==0:
                max_bot_avg_w = height_avg
                bot_group_w = group
                bot_group_side_avg_w = side_avg

        # Process yellow groups
        for group in yellow_seg_groups:
            # Dont care about length analysis, yellow noise can be in huge groups
            side_avg = 0
            height_avg = 0
            for seg in group:
                side_avg = side_avg + seg['y']
                height_avg = height_avg + seg['x']
            side_avg = side_avg/len(group)
            height_avg = height_avg/len(group)

            # Want closest yellow segments to the left
            if (side_avg > -pos_shift_bc_curve and (side_avg < min_left_avg_y or min_left_avg_y < -pos_shift_bc_curve)) or len(left_group_y) == 0:
                min_left_avg_y = side_avg
                left_group_y = group
            if height_avg > max_top_avg_y or len(top_group_y) == 0:
                max_top_avg_y = height_avg
                top_group_y = group
                top_group_side_avg_y = side_avg
            if height_avg < max_bot_avg_y or len(bot_group_y) == 0:
                max_bot_avg_y = height_avg
                bot_group_y = group
                bot_group_side_avg_y = side_avg
        
        # 
        if len(yellow_seg_groups) > 0 and len(white_seg_groups) > 0:
            if max_bot_avg_y > max_top_avg_w and abs(bot_group_side_avg_y - top_group_side_avg_w) < horizontal_distribution:
                yellow_seg_groups = []

        # Branch based on whether we have yellow lane info
        direction = "straight"
        if len(yellow_seg_groups) > 0:
            #print("here")
            # Now see whether we have white groups to work with
            if len(white_seg_groups) > 0:
                max_y_w = -100000
                min_y_w = 100000
                max_y_y = -100000
                min_y_y = 100000
                for seg in right_group_w:
                    if seg['y'] > max_y_w:
                        max_y_w = seg['y']
                    if seg['y'] < min_y_w:
                        min_y_w = seg['y']

                for seg in left_group_y:
                    if seg['y'] > max_y_y:
                        max_y_y = seg['y']
                    if seg['y'] < min_y_y:
                        min_y_y = seg['y']
                possible_turn = False
                if abs(max_y_w - min_y_w) > horizontal_distribution or abs(max_y_y - min_y_y) > horizontal_distribution:
                    possible_turn = True

                # Check if lane appears "normal" in the sense of yellow line to left and white to right
                # If white lane is above yellow and both lanes are to the left then assume left turn    
                if (
                    max_top_avg_w > max_top_avg_y 
                    and top_group_side_avg_y > -pos_shift_bc_curve 
                    and top_group_side_avg_w > -pos_shift_bc_curve
                    and possible_turn
                ):
                    direction = "left"
                    right_white_lane = top_group_w
                    left_yellow_lane = top_group_y
                # If white lane is below yellow and both lanes are to the right then assume right turn
                elif(
                    max_bot_avg_w < max_bot_avg_y 
                    and bot_group_side_avg_w < pos_shift_bc_curve 
                    and bot_group_side_avg_y > bot_group_side_avg_w
                    and possible_turn
                ):
                    direction = "right"
                    right_white_lane = bot_group_w
                    left_yellow_lane = bot_group_y
                elif (min_y_y > max_y_w) and (abs(min_left_avg_y - min_right_avg_w) > pos_shift_bc_curve):
                    direction = "straight"
                    right_white_lane = right_group_w
                    left_yellow_lane = left_group_y
                # Something confusing happening, stick with last confident answer
                else:
                    #print("yellow and white path unknown")
                    direction = "unknown"
                    right_white_lane = []
                    left_yellow_lane = []

            # This path is yellow segments only
            else:
                # Very difficult to make use of top/bottom yellow lines without white for reference
                # Going to use just closest left lane
                # Try to predict direction from horizontal distribution of segments
                max_y = -100000
                min_y = 100000
                for seg in left_group_y:
                    if seg['y'] > max_y:
                        max_y = seg['y']
                    if seg['y'] < min_y:
                        min_y = seg['y']
                # If the closest left lane is actually to the left, could be straight or left turn
                if min_y > -pos_shift_bc_curve and max_y > -pos_shift_bc_curve:
                    if abs(max_y - min_y) > horizontal_distribution:
                        direction = "left"
                        left_yellow_lane = left_group_y
                    else:
                        direction = "straight"
                        left_yellow_lane = left_group_y
                # If closest left lane not entirely to left, could be a right turn
                elif max_y > -pos_shift_bc_curve and abs(max_y - min_y) > horizontal_distribution:
                    direction = "right"
                    left_yellow_lane = left_group_y
                else:
                    #print("yellow path unknown")
                    direction = "unknown"
                    left_yellow_lane = []
        # This path is white segments only
        else:
            # If we have a huge group, assume it is the right one since we have best vision of it
            # This assumption is made bc very little white noise observed... maybe wrong
            if max_len_w > 2*sec_max_len_w:
                right_white_lane = max_group
                
                # Try to predict direction from horizontal distribution of segments
                max_y = -100000
                min_y = 100000
                for seg in max_group:
                    if seg['y'] > max_y:
                        max_y = seg['y']
                    if seg['y'] < min_y:
                        min_y = seg['y']
                if max_y < 0 and min_y < 0:
                    # If group is to the right and has big distribution, assume right turn
                    if abs(max_y - min_y) > horizontal_distribution:
                        direction = "right"
                    else:
                        direction = "straight"
                # If some components are to the left and has big distribution, assume right turn
                elif max_y > 0 and min_y < 0 and abs(max_y - min_y) > horizontal_distribution:
                    direction = "left"
                else:
                    direction = "straight"
            elif bot_group_w != top_group_w:
                max_y = -100000
                min_y = 100000
                for seg in right_group_w:
                    if seg['y'] > max_y:
                        max_y = seg['y']
                    if seg['y'] < min_y:
                        min_y = seg['y']
                # If we observe a logical right group, trust it
                if max_y < pos_shift_bc_curve and abs(max_y - min_y) < horizontal_distribution:
                    right_white_lane = right_group_w
                    direction = "straight"
                # If top group is to the left, assume making left turn
                elif top_group_side_avg_w > 0: 
                    right_white_lane = top_group_w
                    direction = "left"
                # If bot group is to the right, assume making right turn
                elif bot_group_side_avg_w < 0:
                    right_white_lane = bot_group_w
                    direction = "right"
                else:
                    right_white_lane = right_group_w
                    direction = "straight"
            # If we have no idea, assume the one immediately to our right is the one
            else:
                right_white_lane = right_group_w
                direction = "straight"
        return direction