import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import numpy as np


class GridMapGenerator(Node):

    def __init__(self) -> None:
        # Initializing the Node
        super().__init__('occupancy_map_generator')

        # Initializing paramaters from YAML file
        self.declare_parameters(namespace='', parameters=[
            ('resolution', 1.0),
            ('grid_width', 100),
            ('grid_height', 100),
            ('update_delay', 0.5),
            ('point_weight', 13),
            ('decay_rate', 10),
            ('z_filter_height', 1.0)
        ])

        self.grid_res: float = self.get_parameter('resolution').value
        self.grid_width: int = self.get_parameter('grid_width').value
        self.grid_height: int = self.get_parameter('grid_height').value
        self.map_origin: Pose = self.generate_map_origin()

        self.z_filter_height: float = self.get_parameter(
            'z_filter_height').value
        self.point_weight: int = self.get_parameter('point_weight').value
        self.decay_rate: int = self.get_parameter('decay_rate').value

        self.get_logger().info(
            f'point_weight: {self.point_weight}, decay_rate: {self.decay_rate}')

        # Creating Publishers and Subscribers
        self.latest_pointcloud: np.array = None

        self.poincloud_sub = self.create_subscription(
            PointCloud2,
            '/perception/lidar/points_filtered',
            self.pointcloud_callback,
            10
        )

        self.occupancy_map_pub = self.create_publisher(
            OccupancyGrid,
            '/mapping/occupancy_map',
            10
        )

        # Initializing the OccupancyGrid map
        self.occupancy_map = OccupancyGrid(
            header=Header(
                stamp=Time(sec=int(self.get_clock().now().seconds_nanoseconds()[0]),
                           nanosec=int(self.get_clock().now().seconds_nanoseconds()[1])),
                frame_id='map'
            ),

            info=MapMetaData(
                resolution=self.grid_res,
                width=self.grid_width,
                height=self.grid_height,
                origin=self.map_origin
            )
        )

        # Setting map data to 0 (Representing open space)
        self.occupancy_map.data = [0 for _ in range(
            self.grid_height * self.grid_width)]

        # Creating Timer for how often to update the occupancy_map
        self.update_occupancy_map_timer = self.create_timer(
            self.get_parameter('update_delay').value, self.update_occupancy_map
        )

    def generate_map_origin(self) -> Pose:
        delta_width_meters = (self.grid_width * self.grid_res) / 2
        delta_height_meters = (self.grid_height * self.grid_res) / 2

        pose = Pose()

        # Places the map_frame in the center of the gridmap
        pose.position.x = -delta_width_meters
        pose.position.y = -delta_height_meters
        pose.position.z = 0.0

        # Orientation is same as map_frame orientation
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        return pose

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        point_generator = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)

        points = list(point_generator)
        points = [[i[1], i[0], i[2]] for i in points]
        points_array = np.array(points)
        # filtered_points = self.z_filter(points_array, self.z_filter_height)
        transformed_points = self.transform_points(points_array)

        self.latest_pointcloud = np.array(transformed_points)

    def transform_points(self, points: np.array) -> np.array:
        # TODO: Transform the Points properly

        transformed_points = points.copy()
        transformed_points[:, 1] *= -1

        return transformed_points

    def z_filter(self, points: np.array, filter_height: float) -> np.array:
        # self.get_logger().info(points)
        pointcloud_mask = points[:, 2] >= -self.z_filter_height
        return points[pointcloud_mask]

    def update_occupancy_map(self) -> None:

        if self.latest_pointcloud is not None and self.latest_pointcloud.size != 0:
            binned_tiles: np.array = self.bin_points(
                self.latest_pointcloud, self.grid_res)

            curr_scan = np.zeros(
                (self.grid_width * self.grid_height), dtype=np.int8)

            for tile in binned_tiles:
                tile_index = self.grid_width * (tile[0]) + tile[1]

                # Pre-Decay implementation
                # self.occupancy_map.data[tile_index] = 100

                # With Decay
                curr_scan[tile_index] += 1

            for i in range(curr_scan.shape[0]):
                curr_val = self.occupancy_map.data[i]

                if curr_scan[i] > 0:
                    self.occupancy_map.data[i] = min(
                        curr_val + curr_scan[i] * self.point_weight, 100)
                else:
                    self.occupancy_map.data[i] = max(
                        curr_val - self.decay_rate, 0)

        elif self.latest_pointcloud is None:
            self.get_logger().info("No PointCloud2 to process")
        else:
            self.get_logger().info("No points in pointcloud")

        now = self.get_clock().now()
        self.occupancy_map.header.stamp = now.to_msg()
        self.occupancy_map_pub.publish(self.occupancy_map)

    def bin_points(self, points: np.array, resolution: float) -> np.array:

        binned_points = np.floor(points.copy() / resolution)
        # binned_points = np.floor(np.divide(points.copy(), resolution))
        binned_points[:, 0] = binned_points[:, 0] + self.grid_width // 2
        binned_points[:, 1] = self.grid_height // 2 - binned_points[:, 1]

        # Filtering the points to make sure they are in bounds and no repeat tiles
        binned_points = binned_points[np.all((binned_points[:, :2] >= [0, 0]) & (
            binned_points[:, :2] < [self.grid_width, self.grid_height]), axis=1)]

        # binned_points = np.unique(binned_points.astype(np.int32), axis=0)
        binned_points = binned_points.astype(np.int32)

        return binned_points


def main(args=None):
    rclpy.init(args=args)
    node = GridMapGenerator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
