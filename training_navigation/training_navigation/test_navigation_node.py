import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry, Path
from .planner import StraightPlanner, CustomPlanner


class TestNavigationNode(Node):

    def __init__(self):
        super().__init__('test_navigation')

        self.declare_parameters(namespace='', parameters=[
            ('planner', 'STRAIGHT_LINE')
        ])

        self.odom_sub = self.create_subscription(Odometry, '/localization/odometry',
                                                 self.odom_callback, 10)

        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose',
                                                 self.goal_callback, 10)

        ###
        # For trainees:
        # Create a subscriber to /perception/lidar/bounding_boxes
        # to get bounding boxes of obstacles
        ###

        self.path_pub = self.create_publisher(Path, '/navigation/plan', 10)

        self.robot_pose: Pose = None
        self.goal: Pose = None

        # When you change the planner parameter in the yaml, the node
        # will run your implementation
        if self.get_parameter('planner').value == 'STRAIGHT_LINE':
            self.planner = StraightPlanner()
        else:
            self.planner = CustomPlanner()

    def odom_callback(self, msg: Odometry):
        self.robot_pose = msg.pose.pose

    def goal_callback(self, msg: PoseStamped):
        self.goal = msg.pose

        self.create_path()

    def create_path(self):
        if self.robot_pose is None:
            self.get_logger().info('No Robot Pose: Cannot Create Path')
            return

        if self.goal is None:
            self.get_logger().info('No Goal Pose: Cannot Create Path')
            return

        self.planner.robot_pose = self.robot_pose
        self.planner.goal_pose = self.goal

        # For trainees:
        # Pass in the detected obstacles to your implementation of create_plan
        plan = self.planner.create_plan()

        plan.header.frame_id = 'map'

        self.path_pub.publish(plan)


def main(args=None):
    rclpy.init(args=args)

    node = TestNavigationNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
