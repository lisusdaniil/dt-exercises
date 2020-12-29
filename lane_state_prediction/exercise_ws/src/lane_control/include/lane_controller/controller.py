import numpy as np

class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):
        print("init pp")
        print(parameters)
        self.parameters = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters

    def update_target_point(self, prev_target, direction, white_seg_full, yellow_seg_full):
        """

            Args:
                
        """

        look_ahead_dist = self.parameters['~look_ahead_dist']
        correction_from_edge = self.parameters['~correction_from_edge']
        max_yellow_white_dist = self.parameters['~max_yellow_white_dist']

        if len(white_seg_full) == 0 and len(yellow_seg_full) == 0:
            #print("seg_full empty")
            return prev_target.x, prev_target.y

        white_seg_filt = self.filter_segment_list(white_seg_full)
        yellow_seg_filt = self.filter_segment_list(yellow_seg_full)

        if len(white_seg_filt) == 0 and len(yellow_seg_filt) == 0:
            #print("seg_filt empty")
            return prev_target.x, prev_target.y

        # Sort to maybe help grouping, not really sure if this has impact but why not
        white_seg_filt.sort(key=self.sort_by_y)
        # Group together segments to hopefully form lanes
        white_seg_groups = self.group_segs_together(white_seg_filt)
        yellow_seg_groups = self.group_segs_together(yellow_seg_filt)

        if len(white_seg_groups) == 0 and len(yellow_seg_groups)==0:
            #print("seg_groups empty")
            return prev_target.x, prev_target.y

        # Sort in ascending vertical direction
        for group in white_seg_groups:
            group.sort(key=self.sort_by_x)
        for group in yellow_seg_groups:
            group.sort(key=self.sort_by_x)

        # Try to predict which white segment group is the right lane and the direction it is heading in
        # Also predict which yellow segment group is the middle yellow line
        right_white_lane, left_yellow_lane = self.get_lane_groups(white_seg_groups, yellow_seg_groups, direction)

        white_untrustworthy = False
        # Check if we can trust white
        if len(right_white_lane) < 1:
            white_untrustworthy = True

        # Finally, predict the target point
        if not white_untrustworthy:
            # Find closest white point to look ahead distance
            closest_white_seg = self.closest_to_lookahead(look_ahead_dist, right_white_lane, direction)

        yellow_untrustworthy = False
        if len(left_yellow_lane) > 0:
            closest_yellow_seg = self.closest_to_lookahead(look_ahead_dist, left_yellow_lane, direction)

            if not white_untrustworthy:
                # Dont trust white if it is close to the yellow segment and both are on your left
                if (closest_yellow_seg['y'] - closest_white_seg['y']) < 0.05 and closest_white_seg['y'] > 0 :
                    white_untrustworthy = True
                    # Knowing we cant trust white, get new yellow
                    right_white_lane, left_yellow_lane = self.get_lane_groups([], yellow_seg_groups, direction)

            if not white_untrustworthy:
                # Dont trust yellow if it appears on the right of the white line and the white line
                # is in a reasonable spot to your right
                #print(closest_white_seg['y'])
                if closest_yellow_seg['y'] < closest_white_seg['y'] and closest_white_seg['y'] < -0.05 :
                    yellow_untrustworthy = True
                else:
                    if self.meas_dis(closest_yellow_seg, closest_white_seg) > max_yellow_white_dist:
                        yellow_untrustworthy = True
                    else:
                        #print("yellow and white")
                        new_target_x = (closest_white_seg['x'] + closest_yellow_seg['x'])/2
                        new_target_y = (closest_white_seg['y'] + closest_yellow_seg['y'])/2
            else:
                #print("yellow")
                if direction == "straight":
                    new_target_x = closest_yellow_seg['x'] 
                    new_target_y = closest_yellow_seg['y'] - correction_from_edge
                elif direction == "left":
                    new_target_x = closest_yellow_seg['x'] + correction_from_edge/1.3
                    new_target_y = closest_yellow_seg['y'] + correction_from_edge/5
                elif direction == "right":
                    new_target_x = closest_yellow_seg['x'] - correction_from_edge/1.3
                    new_target_y = closest_yellow_seg['y'] - correction_from_edge/5
                else:
                    new_target_x = prev_target.x
                    new_target_y = prev_target.y
        else:
            yellow_untrustworthy = True

        if yellow_untrustworthy:
            if not white_untrustworthy:
                #print("white")
                # No yellow lines to work with, go off of predicted right white lane + direction
                if direction == "straight":
                    new_target_x = closest_white_seg['x']
                    new_target_y = closest_white_seg['y'] + correction_from_edge
                elif direction == "left":
                    new_target_x = closest_white_seg['x'] - correction_from_edge/1.3
                    new_target_y = closest_white_seg['y'] + correction_from_edge/4
                elif direction == "right":
                    new_target_x = closest_white_seg['x'] + correction_from_edge/1.3
                    new_target_y = closest_white_seg['y'] - correction_from_edge/4
            # If we reach here, neither yellow nor white can be trusted
            else:
                #print("neither trustworthy")
                return prev_target.x, prev_target.y

        return new_target_x, new_target_y

    def sort_by_y(self, e):
        return e['y']

    def sort_by_x(self, e):
        return e['x']

    def closest_to_lookahead(self, look_ahead_dist, seg_list, direction):
        min_dist = 10000
        
        car_pos = {"x": 0.0, "y": 0.0}      # Segments are in pos relative to car which is at (0, 0)
        closest_seg = car_pos
        for seg in seg_list:
            dist = self.meas_dis(car_pos, seg)
            if abs(dist-look_ahead_dist) < min_dist:
                # Ignore segments that we know are in the wrong direction even if they are closest
                if direction == 'right' and seg['y'] > 0:
                    continue
                elif direction == 'left' and seg['y'] < 0:
                    continue
                else:
                    min_dist = abs(dist-look_ahead_dist)
                    closest_seg = seg
        return closest_seg

    def get_lane_groups(self, white_seg_groups, yellow_seg_groups, direction):
        horizontal_distribution = self.parameters['~horizontal_distribution']
        pos_shift_bc_curve = self.parameters['~pos_shift_bc_curve']

        right_white_lane = []
        left_yellow_lane = []
        max_len_w = 0
        sec_max_len_w = 0
        min_right_avg_w = 10000
        max_top_avg_w = -1000
        max_bot_avg_w = 10000
        right_group_w = []
        top_group_w = []
        bot_group_w = []

        max_len_y = 0
        sec_max_len_y = 0
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
        
        if len(yellow_seg_groups) > 0 and len(white_seg_groups) > 0:
            if max_bot_avg_y > max_top_avg_w and abs(bot_group_side_avg_y - top_group_side_avg_w) < horizontal_distribution:
                yellow_seg_groups = []

        # Now assign lanes to return
        if direction == "left":
            right_white_lane = top_group_w
            left_yellow_lane = top_group_y
        # If white lane is below yellow and both lanes are to the right then assume right turn
        elif direction == "right":
            right_white_lane = bot_group_w
            left_yellow_lane = bot_group_y
        elif direction == "straight":
            right_white_lane = right_group_w
            left_yellow_lane = left_group_y

        return right_white_lane, left_yellow_lane

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

    def compute_control_action(self, target_point, K):
        """

            Args:
                
        """
        v_nom = self.parameters['~v_nom']
        v_min = self.parameters['~v_min']
        alpha_K = self.parameters['~alpha_K']

        d = max(np.sqrt(target_point.x**2 + target_point.y**2),0.001)
        sin_alpha = target_point.y/d
        alpha = np.arcsin(sin_alpha)
        
        v = v_nom*np.exp(-alpha_K*abs(alpha))
        v = max(v_min, v)
        #v = 0.2
        omega = sin_alpha/K

        return v, omega

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

    def meas_dis(self, seg1, seg2):
        """Calculated the euclidean distance between two single point segments.

            Args:
                seg1 (:obj:`dict`): dictionary containing an x and y coordinate.
                seg2 (:obj:`dict`): dictionary containing an x and y coordinate.
        """

        return ((seg1['x'] - seg2['x'])**2 + (seg1['y'] - seg2['y'])**2)**0.5
