#!/usr/bin/env python3
# Copyright (c) [2022] [Adam Serafin]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from vision_msgs.msg import Detection3DArray, Detection2DArray, Detection2D, BoundingBox2D
from visualization_msgs.msg import ImageMarker, MarkerArray, Marker
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import ColorRGBA, String
from foxglove_msgs.msg import ImageMarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
import copy


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    q[0] = cj * sc - sj * cs
    q[1] = cj * ss + sj * cc
    q[2] = cj * cs - sj * sc
    q[3] = cj * cc + sj * ss

    return q


class ObjectPublisher(Node):
    def __init__(self):
        super().__init__("object_publisher")
        self._sub_ = self.create_subscription(
            Detection3DArray, "/oak/nn/detections", self.publish_data, 10
        )
        self._det_pub = self.create_publisher(
            ImageMarkerArray, "/oak/nn/detection_markers", 10
        )
        self._2Ddet_pub = self.create_publisher(Detection2DArray, 'yolo/detections2d', 10)

        self._text_pub = self.create_publisher(MarkerArray, "/oak/nn/text_markers", 10)
        self._image_sub_ = self.create_subscription(
            Image, "/oak/rgb/image_raw", self.read_image, 10
        )
        self._depth_image_sub_ = self.create_subscription(
            Image, "/oak/stereo/image_raw", self.read_depth_image, 10
        )
        self._image_pub = self.create_publisher(Image, "/oak/rgb/image_bb", 10)
        self._depth_image_pub = self.create_publisher(Image, "/depth_camera/depth/image_rect_raw", 10)

        self._br = TransformBroadcaster(self)
        self._unique_id = 0
        self.br = CvBridge()
        self.img = None
        self.depth_img = None

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Publish static transforms once at startup
        self.make_transforms()

        self.get_logger().info("ObjectPublisher node Up!")

    def make_transforms(self):
        # Publishes a static transform between the torso and the oak-d camera fixed on Reachy's right shoulder.
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "torso"
        t.child_frame_id = "oak-d-base-frame"

        t.transform.translation.x = 0.01
        t.transform.translation.y = -0.1385
        t.transform.translation.z = 0.11
        quat = quaternion_from_euler(0.0, 50.0 * math.pi / 180.0, 0.0)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_static_broadcaster.sendTransform(t)

    def read_image(self, img: Image):
        self.img = self.br.imgmsg_to_cv2(img)
        
    def read_depth_image(self, img: Image):
        self.depth_img = self.br.imgmsg_to_cv2(img)

    def publish_data(self, msg: Detection3DArray):
        markerArray = ImageMarkerArray()
        textMarker = MarkerArray()
        detections2D = Detection2DArray()
        i = 0
        if self._unique_id > 50:
            self._unique_id = 0
        for det in msg.detections:
            bbox = det.bbox
            det.results[0]
            label = f"{det.results[0].hypothesis.class_id}_{i + self._unique_id}"
            det_pose = det.results[0].pose.pose

            textMarker.markers.append(
                Marker(
                    header=msg.header,
                    id=i + self._unique_id,
                    scale=Vector3(x=0.1, y=0.1, z=0.1),
                    type=Marker.TEXT_VIEW_FACING,
                    color=ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0),
                    action=0,
                    text=label,
                    pose=det_pose,
                )
            )

            markerArray.markers.append(
                ImageMarker(
                    header=msg.header,
                    id=i + self._unique_id,
                    scale=1.0,
                    filled=1,
                    type=ImageMarker.LINE_STRIP,
                    outline_color=ColorRGBA(r=0.0, g=255.0, b=255.0, a=255.0),
                    fill_color=ColorRGBA(b=255.0, a=0.2),
                    points=[
                        Point(
                            x=bbox.center.position.x - bbox.size.x / 2,
                            y=bbox.center.position.y + bbox.size.y / 2,
                            z=0.0,
                        ),
                        Point(
                            x=bbox.center.position.x + bbox.size.x / 2,
                            y=bbox.center.position.y + bbox.size.y / 2,
                            z=0.0,
                        ),
                        Point(
                            x=bbox.center.position.x + bbox.size.x / 2,
                            y=bbox.center.position.y - bbox.size.y / 2,
                            z=0.0,
                        ),
                        Point(
                            x=bbox.center.position.x - bbox.size.x / 2,
                            y=bbox.center.position.y - bbox.size.y / 2,
                            z=0.0,
                        ),
                        Point(
                            x=bbox.center.position.x - bbox.size.x / 2,
                            y=bbox.center.position.y + bbox.size.y / 2,
                            z=0.0,
                        ),
                    ],
                )
            )
            # Creating a 2D detection message from the 3D detection message (that contains the 2D bouding box...)
            # TODO check if normalized...
            bbox2D = BoundingBox2D()
            bbox2D.center.position.x = bbox.center.position.x
            bbox2D.center.position.y = bbox.center.position.y
            bbox2D.size_x = bbox.size.x
            bbox2D.size_y = bbox.size.y
            detections2D.detections.append(
                Detection2D(
                    header=msg.header,
                    results=det.results,
                    bbox=bbox2D,
                    id=det.id,
                )
            )
            

            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()

            tf.child_frame_id = label
            tf.header.frame_id = msg.header.frame_id
            tf.transform.translation.x = det.results[0].pose.pose.position.x
            tf.transform.translation.y = det.results[0].pose.pose.position.y
            tf.transform.translation.z = det.results[0].pose.pose.position.z
            self._br.sendTransform(tf)
            i += 1
            self._unique_id += 1

        if self.img is not None:
            img_with_bb = copy.deepcopy(self.img)
            self.draw_bounding_boxes(img_with_bb, msg)
            cv2.imshow("/oak/rgb/image_bb", img_with_bb)
            cv2.waitKey(1)
            self._image_pub.publish(self.br.cv2_to_imgmsg(img_with_bb))
            
        if self.depth_img is not None:
            depth_img = copy.deepcopy(self.depth_img)
            # self.get_logger().warning(f"dtype {depth_img.dtype}, shape {depth_img.shape}")
            # self.get_logger().warning(f"max value = {np.amax(depth_img)}")
            # self.get_logger().warning(f"min value = {np.amin(depth_img)}")
            
            # TODO double check this... The result should be in meters
            depth_img_float = (depth_img * 255.0 / (65536*10)).astype(np.float32)
            depth_img_float_msg = self.br.cv2_to_imgmsg(depth_img_float, '32FC1')
            # Creating a depth image in the same format that the one used by Gazebo
            self._depth_image_pub.publish(depth_img_float_msg)
            
            # depth_array = np.array(depth_img, dtype=np.int16)
            # depth_norm = depth_array/np.ma.masked_invalid(depth_array).max()*255
            
            # self.get_logger().warning(f"norm max value = {np.amax(depth_norm)}")
            # self.get_logger().warning(f"norm min value = {np.amin(depth_norm)}")
            
            # With this normalization, the pixel values of Red are cm...
            depth_img = (10*depth_img * 255.0 / 65536).astype(np.uint8)
            depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_HOT)
            depth_img = np.ascontiguousarray(depth_img)
            # self.get_logger().warning(f"new max value = {np.amax(depth_img)}")
            # self.get_logger().warning(f"new min value = {np.amin(depth_img)}")
            # self.get_logger().warning(f"new dtype {depth_img.dtype}, shape {depth_img.shape}")
            
            cv2.imshow("/oak/stereo/image_raw (colored)", depth_img)
            cv2.imshow("/depth_camera/depth/image_rect_raw (float)", depth_img_float)
            
            # cv2.imshow("depth_norm", depth_norm)
            cv2.waitKey(1)
            
        if self.img is not None and self.depth_img is not None:
            # Need to have both frames in BGR format before blending
            # if len(frameDisp.shape) < 3:
            #     frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(self.img , 0.5, depth_img, 0.5, 0)
            cv2.imshow("blended", blended)

        self._det_pub.publish(markerArray)
        self._text_pub.publish(textMarker)
        self._2Ddet_pub.publish(detections2D)

    def draw_bounding_boxes(
        self, im: np.ndarray, detection3D: Detection3DArray
    ) -> np.ndarray:
        for d in detection3D.detections:
            nb_results = len(d.results)
            if nb_results < 1:
                self.get_logger().warning("0 results for detection (should be 1)")
                continue
            elif nb_results > 1:
                self.get_logger().warning(
                    f"{nb_results} results for detection (should be 1, check this)"
                )

            result = d.results[0]
            id = result.hypothesis.class_id
            score = result.hypothesis.score
            xyz = result.pose.pose.position

            # The 3D bbox message is used to transmit the 2D bbox instead
            # TODO this is an ugly and non exact fix, with the assumption that the magic factor is 740/640
            # I posted an issue on this here:
            # https://github.com/luxonis/depthai-ros/issues/259
            magic_factor = 1.15625
            d.bbox.size.x *= magic_factor
            d.bbox.size.y *= magic_factor
            d.bbox.center.position.x *= magic_factor
            d.bbox.center.position.y *= magic_factor

            size_x_2 = d.bbox.size.x // 2
            size_y_2 = d.bbox.size.y // 2
            d.bbox.center.position.x += 275
            cv2.rectangle(
                im,
                (
                    (int)(d.bbox.center.position.x - size_x_2),
                    (int)(d.bbox.center.position.y - size_y_2),
                ),
                (
                    (int)(d.bbox.center.position.x + size_x_2),
                    (int)(d.bbox.center.position.y + size_y_2),
                ),
                (255, 0, 0),
                4,
            )

            font_scale = d.bbox.size.x / 1080.0 * 5
            cv_text = f"id: {id}\nscore: {score:.2f}\nx: {xyz.x:.2f}\ny: {xyz.y:.2f}\nz: {xyz.z:.2f})"
            for i, line in enumerate(cv_text.split("\n")):
                y_offset = i * 30
                cv2.putText(
                    im,
                    line,
                    (
                        (int)(d.bbox.center.position.x - size_x_2),
                        (int)(d.bbox.center.position.y - size_y_2 - 5) + y_offset,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        return im


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ObjectPublisher()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
