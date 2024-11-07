#!/usr/bin/env python3

import rospy
import cv2
import gi
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rsun_video_streaming.msg import StreamInfoArray
from gi.repository import Gst

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)


class VideoStreamClient:
    def __init__(self):
        rospy.init_node('gstreamer_client')
        rospy.loginfo("[VideoStreamClient] Initializing Node!")
        self.bridge = CvBridge()

        # Subscribe to the StreamInfoArray message
        rospy.Subscriber("/stream_info", StreamInfoArray, self.stream_info_callback)

        # Dictionary to hold pipeline, publisher, and timer for each stream
        self.streams = {}

    def stream_info_callback(self, msg):
        for stream_info in msg.streams:
            topic = stream_info.topic_name
            udp_port = stream_info.udp_port
            fps = stream_info.fps

            # Create a new stream if it's not already in the streams dictionary
            if topic not in self.streams:
                publisher = rospy.Publisher(f"{topic}/streamed", Image, queue_size=1)
                pipeline, appsink = self.create_pipeline(udp_port)

                # Start a ROS timer to pull frames at the specified FPS
                timer_period = 1.0 / fps
                timer = rospy.Timer(
                    rospy.Duration(timer_period),
                    lambda event, data=(appsink, publisher): self.timer_callback(data)
                )
                self.streams[topic] = {"pipeline": pipeline, "timer": timer, "publisher": publisher}

                rospy.loginfo(f"[VideoStreamClient] Publishing stream for topic {topic}/streamed at {fps} FPS")

    def create_pipeline(self, udp_port):
        # Set up the GStreamer pipeline for receiving the UDP stream
        pipeline = Gst.parse_launch(
            f"udpsrc port={udp_port} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, "
            f"encoding-name=(string)H264, payload=(int)96\" ! rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink name=appsink"
        )
        appsink = pipeline.get_by_name("appsink")
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)

        # Start the pipeline
        pipeline.set_state(Gst.State.PLAYING)
        return pipeline, appsink

    def timer_callback(self, data):
        appsink, publisher = data

        # Try to pull a sample (frame) from the appsink
        sample = appsink.emit("try-pull-sample", 0)
        if sample:
            buf = sample.get_buffer()
            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if success:
                # Retrieve the frame dimensions and format from caps
                caps = sample.get_caps()
                width = caps.get_structure(0).get_value("width")
                height = caps.get_structure(0).get_value("height")

                # Convert buffer to numpy array and reshape to match frame dimensions
                img_array = np.frombuffer(mapinfo.data, dtype=np.uint8)
                img_array = img_array.reshape((height, width, 3))  # Assume BGR format from pipeline

                # Convert numpy array to ROS Image message
                ros_image = self.bridge.cv2_to_imgmsg(img_array, encoding="bgr8")
                ros_image.header.stamp = rospy.Time.now()
                publisher.publish(ros_image)

                buf.unmap(mapinfo)

    def cleanup(self):
        # Set pipelines to NULL state and stop all timers
        for topic, stream_data in self.streams.items():
            stream_data["pipeline"].set_state(Gst.State.NULL)
            stream_data["timer"].shutdown()


if __name__ == '__main__':
    client = VideoStreamClient()
    rospy.on_shutdown(client.cleanup)
    rospy.spin()
