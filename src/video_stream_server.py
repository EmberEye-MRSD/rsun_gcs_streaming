#!/usr/bin/env python3

import rospy
import cv2
import gi
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rsun_video_streaming.msg import StreamInfoArray, StreamInfo

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)


class VideoStreamServer:
    def __init__(self):
        rospy.init_node('gstreamer_node')

        rospy.loginfo("[VideoStreamServer] Initializing Node!")

        self.bridge = CvBridge()
        self.stream_info_pub = rospy.Publisher("/stream_info", StreamInfoArray, queue_size=1)

        # Retrieve topics and setup pipelines
        topics = rospy.get_param("~image_topics", ['/flir_boson/image_raw',
                                                   '/camera/color/image_raw',
                                                   '/camera/infra1/image_rect_raw'])

        text_overlays = rospy.get_param("~text_overlays", ['FLIR Boson',
                                                           'RealSense Color',
                                                           'RealSense Infra1'])

        self.topic_overlays = dict(zip(topics, text_overlays))

        udp_base_port = rospy.get_param("~udp_base_port", 5000)
        udp_host = rospy.get_param("~udp_host", "127.0.0.1")
        fps = rospy.get_param("~fps", 30)  # Assuming default FPS if not provided

        self.subscribers = []
        self.pipelines = {}
        self.stream_infos = []

        for i, topic in enumerate(topics):
            udp_port = udp_base_port + i

            # Add to stream info list with placeholders for frame size and encoding type
            stream_info = StreamInfo()
            stream_info.topic_name = topic
            stream_info.udp_address = udp_host
            stream_info.udp_port = udp_port
            stream_info.fps = fps
            self.stream_infos.append(stream_info)

            # Setup subscriber
            sub = rospy.Subscriber(topic, Image, self.image_callback, callback_args=(topic, udp_host, udp_port, fps))
            self.subscribers.append(sub)

            rospy.loginfo(f"[VideoStreamServer] Streaming topic {topic} with UDP port {udp_port}")

        # Publish StreamInfoArray message
        rospy.Timer(rospy.Duration(1.0), self.publish_stream_info)

    def create_pipeline(self, udp_host, udp_port, width, height, fps, overlay_text=""):
        pipeline = Gst.parse_launch(
            f"appsrc name=source is-live=true do-timestamp=true ! "
            f"videoconvert ! textoverlay text='{overlay_text}' valignment=top halignment=center font-desc='Sans, 24' ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast bitrate=5000 key-int-max=15 ! "
            f"rtph264pay config-interval=1 pt=96 ! udpsink host={udp_host} port={udp_port}"
        )
        appsrc = pipeline.get_by_name("source")
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("block", True)
        appsrc.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw, format=BGR, width={width}, height={height}, framerate={fps}/1"
        ))
        pipeline.set_state(Gst.State.PLAYING)
        return appsrc

    def hist_99(self, image):
        im_srt = np.sort(image.reshape(-1))
        upper_bound = im_srt[round(len(im_srt) * 0.99) - 1]
        lower_bound = im_srt[round(len(im_srt) * 0.01)]
        img = image
        img[img < lower_bound] = lower_bound
        img[img > upper_bound] = upper_bound
        image_out = ((img - lower_bound) / (upper_bound - lower_bound)) * 255.0
        image_out = image_out.astype(np.uint8)
        return image_out

    def publish_stream_info(self, event):
        # Publish the StreamInfoArray
        stream_info_array = StreamInfoArray()
        stream_info_array.streams = self.stream_infos
        self.stream_info_pub.publish(stream_info_array)

    def image_callback(self, msg, args):
        topic, udp_host, udp_port, fps = args

        try:
            # Convert ROS Image message to OpenCV format with appropriate encoding
            encoding = msg.encoding
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)

            # Encoding specific conversions
            if encoding == "mono16":
                cv_image = self.hist_99(cv_image)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                encoding = "bgr8"
            if encoding == "rgb8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                encoding = "bgr8"
            if encoding == "mono8":
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                encoding = "bgr8"

            # Set stream info details based on received image properties
            for stream_info in self.stream_infos:
                if stream_info.topic_name == topic:
                    stream_info.width = msg.width
                    stream_info.height = msg.height
                    stream_info.encoding = encoding

            overlay_text = self.topic_overlays.get(topic, "")

            # Create pipeline if it does not exist yet
            if topic not in self.pipelines:
                appsrc = self.create_pipeline(udp_host, udp_port, msg.width, msg.height, fps, overlay_text)
                self.pipelines[topic] = appsrc
            else:
                appsrc = self.pipelines[topic]

            # Convert OpenCV image to GStreamer compatible format (raw BGR buffer)
            data = cv_image.tobytes()
            buf = Gst.Buffer.new_wrapped(data)
            appsrc.emit("push-buffer", buf)
        except Exception as e:
            rospy.logwarn(f"Failed to process image for topic {topic}: {e}")
            exit(1)

    def cleanup(self):
        # Set pipelines to NULL state
        for pipeline in self.pipelines.values():
            pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    node = VideoStreamServer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.cleanup()
