import rclpy
from rclpy.node import Node
import numpy as np
import math

from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2

from enum import Enum

###

# For trainees:
# Implement the CustomPlanner class at the bottom of the file
# Feel free to change other files as well, though you shouldn't need to make a lot of changes

###

# Constants
MAX_INT32 = np.iinfo(np.int32).max


class Planner:

    def __init__(self):
        self.robot_pose: Pose = None
        self.goal_pose: Pose = None

    # Helper method to create a deep copy of a Pose
    def pose_deep_copy(pose: Pose):
        copy = Pose()
        copy.position.x = pose.position.x
        copy.position.y = pose.position.y
        copy.position.z = pose.position.z
        copy.orientation.x = pose.orientation.x
        copy.orientation.y = pose.orientation.y
        copy.orientation.z = pose.orientation.z
        copy.orientation.w = pose.orientation.w

        return copy


class StraightPlanner(Planner):

    def create_plan(self):
        if self.robot_pose is None or self.goal_pose is None:
            return Path()

        if self.goal_pose.position.x - self.robot_pose.position.x > 0:
            x_dir = 1
        else:
            x_dir = -1

        if self.goal_pose.position.y - self.robot_pose.position.y > 0:
            y_dir = 1
        else:
            y_dir = -1

        path = Path()
        path.header.frame_id = 'map'

        if self.goal_pose.position.x - self.robot_pose.position.x == 0.0:
            theta = .5 * math.pi
        else:
            theta = math.atan(abs(
                (self.goal_pose.position.y - self.robot_pose.position.y) /
                (self.goal_pose.position.x - self.robot_pose.position.x)
            ))

        curr_pose = Planner.pose_deep_copy(self.robot_pose)

        while (
            (abs(curr_pose.position.x - self.robot_pose.position.x)
                < abs(self.goal_pose.position.x - self.robot_pose.position.x)) and
            (abs(curr_pose.position.y - self.robot_pose.position.y)
                < abs(self.goal_pose.position.y - self.robot_pose.position.y))
        ):
            path.poses.append(PoseStamped(
                pose=Planner.pose_deep_copy(curr_pose)))

            curr_pose.position.x += math.cos(theta) * 0.2 * x_dir
            curr_pose.position.y += math.sin(theta) * 0.2 * y_dir

        path.poses.append(PoseStamped(pose=self.goal_pose))

        return path


# For you to implement!
class GridType(Enum):
    EMPTY_SPACE = 0
    OBSTACLE = -1
    OPEN_SET = 1
    CLOSED_SET = 2


class PointCloudProcessor(Node):

    def __init__(self) -> None:
        super().__init__('planner_pointcloud_processor')

        self.latest_pointcloud = None

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/perception/lidar/points_filtered',
            self.pointcloud_callback,
            10
        )

    def pointcloud_callback(self, msg) -> None:
        self.latest_pointcloud = msg
        self.get_logger().info('Received PointCloud2 message!')


class CustomPlanner(Node, Planner):

    def __init__(self, cost_map_size: tuple[int, int]):
        super().__init__('a_star_planner')
        Planner().__init__()

        self.binary_cost_map = np.zeros(cost_map_size, dtype=np.int8)
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            '/perception/lidar/points_filtered',
            self.point_cloud_callback,
            10
        )

    def create_plan(self):
        path = Path()
        path.header.frame_id = 'map'

    @staticmethod
    def euclidian_distance(start_pos: tuple[int, int], end_pos: tuple[int, int]) -> float:
        """
        Gives the normal 2d Euclidian distance between 2 points
        """

        a = end_pos[0] - start_pos[0]
        b = end_pos[1] - start_pos[1]

        return np.sqrt(a*a + b*b)

    # Originally wrote this in a more funcional programming way as I think it makes
    # more sense for this context, hense the @staticmethods
    @staticmethod
    def a_star(cost_map: np.array, start_pos: tuple[int, int], end_pos: tuple[int, int],
               h_func: callable = euclidian_distance) -> list[tuple[int, int]]:

        open_close_set = np.full(cost_map.shape, np.NaN)
        open_close_set[start_pos] = GridType.OPEN_SET

        came_from = np.full(cost_map.shape, np.NaN, dtype=np.int32)

        g_score = np.full(cost_map.shape, MAX_INT32, dtype=np.int32)
        g_score[start_pos] = 0

        f_score = np.full(cost_map.shape, np.Inf)
        f_score[start_pos] = h_func(start_pos, end_pos)

        while np.any(open_close_set == GridType.OPEN_SET):
            f_mask = f_score.copy()
            f_mask[open_close_set != 1] = np.Inf

            current = np.unravel_index(np.argmin(f_mask), cost_map.shape)

            if current == end_pos:
                path_waypoints = CustomPlanner.reconstruct_path(
                    came_from, start_pos, end_pos, cost_map.shape)
                return path_waypoints

            open_close_set[current] = GridType.CLOSED_SET

            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbor = (current[0] + i, current[1] + j)

                    if (i == 0 and j == 0) or not CustomPlanner.is_valid_neighbor(cost_map, neighbor):
                        continue

                    # Dist of 1 if directly next to and dist of sqrt(2) if diagonal
                    grid_dist = 1 if (i == 0 or j == 0) else np.sqrt(2)
                    temp_g = g_score[current] + grid_dist

                    if temp_g < g_score[neighbor]:
                        came_from[neighbor] = np.ravel_multi_index(
                            current, cost_map.shape)

                        g_score[neighbor] = temp_g

                        f_score[neighbor] = temp_g + h_func(neighbor, end_pos)

                        open_close_set[neighbor] = 1

        # If there are no more OPEN_SPACE grids left to check (meaning no valid path)
        # the function retruns a path of None
        return None

    @staticmethod
    def is_valid_neighbor(cost_map: np.array, neighbor: tuple[int, int]) -> bool:
        for i in range(len(neighbor)):
            if neighbor[i] < 0 or neighbor[i] >= cost_map.shape[i]:
                return False

        if cost_map[neighbor] == GridType.OBSTACLE:
            return False

        return True

    @staticmethod
    def reconstruct_path(came_from: np.array, start_pos: tuple[int, int],
                         end_pos: tuple[int, int], grid_shape: tuple[int, int]) -> list[tuple[int, int]]:

        current = end_pos
        path = [current]

        while path[-1] != start_pos:
            current = np.unravel_index(came_from[current], grid_shape)
            path.append(current)

        return path


def main(args=None):
    rclpy.init(args=args)

    pointcloud_processor_node = PointCloudProcessor()

    rclpy.spin(pointcloud_processor_node)
