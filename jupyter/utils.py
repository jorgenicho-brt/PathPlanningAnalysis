import sys
import utm
import os
import copy
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

import context
import PyPL

class Box2D:    
    def __init__(self, points: list[np.array], **kwargs):
        self.points = points
        self.plot_attributes = {'edgecolor':'purple',
                               'linewidth': 3,
                               'facecolor': 'none'}
        for k in self.plot_attributes:
            if k in kwargs:
                self.plot_attributes[k] = kwargs[k]

    @property
    def top_left(self):
        return self.points[0]

    @property
    def top_right(self):
        return self.points[1]

    @property
    def bottom_right(self):
        return self.points[2]

    @property
    def bottom_left(self):
        return self.points[3]

    def plot(self, axs, offset = np.asarray([0.0, 0.0]), **kwargs):
        points = np.stack(self.points, axis = 0)
        points -= offset
        plot_attributes = copy.deepcopy(self.plot_attributes)
        for k in plot_attributes:
            if k in kwargs:
                plot_attributes[k] = kwargs[k]            
        axs.fill(points[:,0], points[:,1],**plot_attributes)

    @classmethod
    def create_box(cls, center: list[float], w: float, h: float, **kwargs):
        '''
        Args:
            kwargs: Display attributes to be forwarded to matplotlib
        '''
        center = np.asarray(center)
        top_left = center + np.asarray([-0.5 * w, 0.5 * h])
        top_right = center + np.asarray([0.5 * w, 0.5 * h])
        bottom_left = center + np.asarray([-0.5 * w, -0.5 * h])
        bottom_right = center + np.asarray([0.5 * w, -0.5 * h])
        return Box2D([top_left, top_right, bottom_right, bottom_left], **kwargs)

    @classmethod
    def compute_bounding_box(cls, boxes: list, padding = [0.0, 0.0], **kwargs):
        '''
        Args:
            kwargs: Display attributes to be forwarded to matplotlib
        '''
        boxes_points = [box.points for box in boxes]
        boxes_points = np.concatenate(boxes_points, axis = 0)
        min_x = np.min(boxes_points[:,0])
        max_x = np.max(boxes_points[:,0])
        min_y = np.min(boxes_points[:,1])
        max_y = np.max(boxes_points[:,1])
        top_left = np.asarray([min_x - padding[0], max_y + padding[1]])
        top_right = np.asarray([max_x + padding[0], max_y + padding[1]])
        bottom_left = np.asarray([min_x - padding[0], min_y - padding[1]])
        bottom_right = np.asarray([max_x + padding[0], min_y - padding[1]])
        return Box2D([top_left, top_right, bottom_right, bottom_left], **kwargs)

def xy_to_pypl_point(p: np.array, zone_number, zone_letter):
    lat, lon = utm.to_latlon(p[0], p[1], zone_number, zone_letter)
    return PyPL.WorldPoint(lat, lon)

def box2d_to_pypl_points(boxes : list[Box2D], zone_number: int, zone_letter: str):
    pypl_boundaries = []
    for box in boxes:
        boundary = []
        for p in box.points:
            lat, lon = utm.to_latlon(p[0], p[1], zone_number, zone_letter)
            boundary.append(PyPL.WorldPoint(lat, lon))
        boundary.append(boundary[0])
        pypl_boundaries.append(boundary)
    return pypl_boundaries

class Pose2D:
    def __init__(self, xy_coords: np.array, angle: float, zone_number, zone_letter):
        self.xy_coords = xy_coords
        self.angle = angle
        self.latlon_coords = xy_to_pypl_point(self.xy_coords, zone_number, zone_letter)
        
    def plot(self, axs, offset = np.asarray([0.0, 0.0]), scale = 1.0, facecolor='purple' ):
        xy_coords = self.xy_coords - offset
        xy_coords = np.resize(xy_coords, 3)
        xy_coords[2] = 0.0
        triangle_points = []
        triangle_points.append(scale * np.asarray([0.5,0.0, 0.0]))
        triangle_points.append(scale * np.asarray([-0.5,0.5, 0.0]))
        triangle_points.append(scale * np.asarray([-0.5,-0.5, 0.0]))
        rotation = R.from_euler('z', self.angle, degrees=True)
        triangle_points = np.stack(triangle_points, axis = 0)
        triangle_points = rotation.apply(triangle_points)
        triangle_points += xy_coords
        axs.fill(triangle_points[:,0], triangle_points[:,1], facecolor = facecolor)

@dataclass
class Tracks:
    def __init__(self, tracks_list: list[PyPL.Track], origin_xy, **kwargs):
        self.tracks_list = tracks_list
        self.origin_xy = origin_xy
        self.plot_attributes = {'color':'lightgreen',
                               'linewidth': 2,
                               'linestyle': '--',
                               'marker': ''}
        for k in self.plot_attributes:
            if k in kwargs:
                self.plot_attributes[k] = kwargs[k]

    def plot(self, axs, **kwargs):
        plot_attributes = copy.deepcopy(self.plot_attributes)
        for k in plot_attributes:
            if k in kwargs:
                plot_attributes[k] = kwargs[k] 
        for i, track in enumerate(self.tracks_list):
            shape_points = []
            for p in track.Shape:
                x, y, zn, zl = utm.from_latlon(p.Lat, p.Lon)
                shape_points.append(np.asarray([x, y]))
                
            shape_points = np.stack(shape_points, axis = 0)
            shape_points = shape_points - self.origin_xy
            axs.plot(shape_points[:,0], shape_points[:,1], **plot_attributes)   

            for i in range(1,shape_points.shape[0]):
                start_p = shape_points[i-1,:]
                end_p = shape_points[i,:]
                dir = end_p - start_p
                arrow_width = 0.5
                arrow_length = 0.5
                color = plot_attributes.get('color', 'green')
                unit_vec = dir/np.linalg.norm(dir)
                dir_vec = arrow_length *unit_vec
                end_p -= dir_vec
                axs.arrow(end_p[0], end_p[1], dir_vec[0], dir_vec[1], color = color, width = arrow_width)

class PlannedPath:
    def __init__(self, name: str, path_points: list[PyPL.WorkPath], valid: bool, **plot_attributes):
        self.name = name
        self.path_points : list[PyPL.Pose2D] = []                            
        self.valid = valid
        self.plot_attributes = {'color':'blue',
                       'linewidth': 4,
                       'linestyle': '-',
                       'marker': '+',
                       'markersize' : 10,
                       'alpha' : 1.0}
        for k in self.plot_attributes:
            if k in plot_attributes:
                self.plot_attributes[k] = plot_attributes[k]

        if len(path_points) == 0:
            logging.warning('Path "{}" is empty'.format(self.name))
        
        for workpath in path_points:
            for path_segment in workpath.PathSegments:
                if len(path_segment.Points) == 0:
                    continue
                for point in path_segment.Points:
                    self.path_points.append(point)
        if len(self.path_points) == 0:
            logging.warning('No points were found in Path "{}"'.format(self.name))

    def get_points_xy(self, offset : np.array = np.asarray([0.0, 0.0])):
        xcoord = [coord.Point.E for coord in self.path_points]
        ycoord = [coord.Point.N for coord in self.path_points]
        output_xy = np.stack([xcoord, ycoord], axis = 1)
        output_xy += offset
        return output_xy   

    def plot(self, axs, offset: np.array, **plot_attributes_modifiers):
        plot_attributes = copy.deepcopy(self.plot_attributes)
        for k in plot_attributes:
            if k in plot_attributes_modifiers:
                plot_attributes[k] = plot_attributes_modifiers[k] 
        if not self.valid:
            plot_attributes['linestyle'] = 'dotted'
            plot_attributes['marker'] = 'x'
            plot_attributes['color'] = 'red'
                
        output_xy = self.get_points_xy(offset)
        axs.plot(output_xy[:,0], output_xy[:,1],**plot_attributes)


@dataclass
class EnvironmentObjects:
    large_obstacles: list[Box2D]
    external_boundaries: list[Box2D] 
    headland_boundaries: list[Box2D]
            
@dataclass
class PlanEnvironmentConstraints:
    origin_xy: np.array
    environment_objects: EnvironmentObjects
    obstacles_buffer: float
    environment_inputs: PyPL.AutoPathBoundariesInputs

@dataclass
class PlanResults:
    start_pose_2d: Pose2D
    end_pose_2d: Pose2D
    implement_path: PlannedPath
    tractor_path: PlannedPath
    tracks:  Tracks

def build_planning_env(origin_xy: np.array, zone_number: int, zone_letter: str,
                       environment_objects: EnvironmentObjects,
                       track_spacing = float(10.0),
                       in_ground_turning_radius = 8.0,
                       vehicle_width = 8.0,
                       implement_width = 8.0,
                       work_heading = 90.0,
                       planning_strategy: PyPL.PlanningStrategy = PyPL.PlanningStrategy.ExistingTrack,
                       obstacles_buffer = 2.0):    
    
    environment_inputs = PyPL.AutoPathBoundariesInputs()
    environment_inputs.TrackSpacing = track_spacing
    environment_inputs.VehicleWidth = vehicle_width
    environment_inputs.ImplementWidth = implement_width
    environment_inputs.WorkingWidth = 1.0 * max([implement_width, vehicle_width])

    environment_inputs.InGroundTurnRadius = in_ground_turning_radius
    
    environment_inputs.Strategy = planning_strategy
    ''' Options for environment_inputs.Strategy are listed below
        SnapToBoundary,
        BestFitSnapToBoundary,
        HeadingOnly,
        ExistingTrack
    '''
    
    environment_inputs.WorkHeading = work_heading
    environment_inputs.InFieldShift = 0.0
    environment_inputs.HeadlandShift = 0.0
    environment_inputs.ExtensionLength = 0.0

    environment_inputs.TopBottom = False
    environment_inputs.TopOffset = 0.0
    environment_inputs.BottomOffset = 0.0
    
    environment_inputs.ExteriorHeadlandOffset = 0.0 
    environment_inputs.InteriorHeadlandOffset = 0.0 
    
    environment_inputs.ExteriorBoundaries = box2d_to_pypl_points(environment_objects.external_boundaries, zone_number, zone_letter)
    environment_inputs.HeadlandBoundaries = box2d_to_pypl_points(environment_objects.headland_boundaries, zone_number, zone_letter)    
    #environment_inputs.SmallImpassables = box2d_to_pypl_points(obstacles, zone_number, zone_letter)
    environment_inputs.LargeImpassables = box2d_to_pypl_points(environment_objects.large_obstacles, zone_number, zone_letter)

    return PlanEnvironmentConstraints(origin_xy, environment_objects, obstacles_buffer, environment_inputs)


def generate_open_field_path(path_end_points : tuple[Pose2D, Pose2D], environment_constraints: PlanEnvironmentConstraints,
                             field_partition_is_impassable = False):

    path_inputs = PyPL.PointToPointInputs()
    path_inputs.StartPose = path_end_points[0].latlon_coords
    path_inputs.StartAngle = path_end_points[0].angle
    path_inputs.EndPose =  path_end_points[1].latlon_coords
    path_inputs.EndAngle = path_end_points[1].angle
    path_inputs.ExtraBuffer = environment_constraints.obstacles_buffer
    path_inputs.FieldPartitionIsImpassable = field_partition_is_impassable

    wrapper = PyPL.PointToPointWrapper() 
    wrapper.SetInputs(environment_constraints.environment_inputs) 
    tracks_list = wrapper.Plan()
    plan_output: PyPL.PlanOutput = wrapper.CreatePttoPt_OpenField(path_inputs) 
    valid = True if plan_output.error_result.Code == 0 else False
    if not valid:
        logging.error('Planning Failed, error msg: {}'.format(plan_output.error_result.Msg))

    origin_xy = environment_constraints.origin_xy
    return PlanResults(path_end_points[0], path_end_points[1], PlannedPath('Implement', plan_output.ImplementPath, valid),
                       PlannedPath('Tractor', plan_output.TractorPath, valid, color = 'gray', marker = '+', linewidth = 2, markersize = 20, alpha= 0.5),
                       Tracks(tracks_list, origin_xy))

def generate_road_network_path(path_end_points : tuple[Pose2D, Pose2D], environment_constraints: PlanEnvironmentConstraints):

    path_inputs = PyPL.PointToPointInputs()
    path_inputs.StartPose = path_end_points[0].latlon_coords
    path_inputs.StartAngle = path_end_points[0].angle
    path_inputs.EndPose =  path_end_points[1].latlon_coords
    path_inputs.EndAngle = path_end_points[1].angle
    path_inputs.ExtraBuffer = environment_constraints.obstacles_buffer

    wrapper = PyPL.PointToPointWrapper() 
    wrapper.SetInputs(environment_constraints.environment_inputs) 
    tracks_list = wrapper.Plan()
    plan_output = wrapper.CreatePttoPt_RoadNet(path_inputs) 

    valid = True if plan_output.error_result.Code == 0 else False
    if not valid:
        logging.error('Planning Failed, error msg: {}'.format(plan_output.error_result.Msg))

    origin_xy = environment_constraints.origin_xy
    return PlanResults(path_end_points[0], path_end_points[1], PlannedPath('Implement', plan_output.ImplementPath, valid),
                       PlannedPath('Tractor', plan_output.TractorPath, valid, color = 'gray', marker = '+', linewidth = 2, markersize = 20, alpha= 0.5),
                       Tracks(tracks_list, origin_xy))

def plot_planning_results(plan_env_constraints: PlanEnvironmentConstraints,
                          plan_results: PlanResults,
                          output_path_offset : np.array,
                          grid_range: list,
                          start_circle_radius = 8.0,
                          end_circle_radius = 8.0):
    fig, axs = plt.subplots(1, 1, layout = 'constrained')
    if not isinstance(axs, (list, tuple, np.ndarray)):
        axs = [axs]
    fig.set_figheight(20)
    fig.set_figwidth(20)
    fig.suptitle('Path Planning Results')
    
    origin_xy = plan_env_constraints.origin_xy
    plan_results.tracks.plot(axs[0])
        
    # Plot start and end points
    start_xy = plan_results.start_pose_2d.xy_coords
    end_xy = plan_results.end_pose_2d.xy_coords
    
    plan_results.start_pose_2d.plot(axs[0], origin_xy, start_circle_radius, facecolor = 'green')
    plan_results.end_pose_2d.plot(axs[0], origin_xy, end_circle_radius, facecolor = 'purple')
    
    # Plot path
    plan_results.implement_path.plot(axs[0], output_path_offset)
    plan_results.tractor_path.plot(axs[0], output_path_offset)
    
    # Plotting boxes
    for box in plan_env_constraints.environment_objects.large_obstacles:
        box.plot(axs[0], origin_xy)
        #break
    for box in plan_env_constraints.environment_objects.external_boundaries:
        box.plot(axs[0], origin_xy, edgecolor = 'brown')

    for box in plan_env_constraints.environment_objects.headland_boundaries:
        box.plot(axs[0], origin_xy, edgecolor = 'orange')
    
    axs[0].set_xlabel('E meters')
    axs[0].set_ylabel('N meters')
    
    # setting up grid
    major_ticks = grid_range #
    axs[0].set_xticks(major_ticks)
    axs[0].set_yticks(major_ticks)
    axs[0].grid(color = 'gray', linestyle = '--', linewidth = 1.0, alpha = 0.8)

    return fig, axs