import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GLib

import sys
import os
import pyds
import time
import threading
import numpy as np

# Paths
YOLO_CONFIG_PATH = "/home/piyansh/DeepStream-Yolo/config_infer_primary_yoloV5.txt"
YOLO_ENGINE_PATH = "/home/piyansh/DeepStream-Yolo/model_b1_gpu0_fp32.engine"
VIDEO_FILES = [
   "/home/piyansh/multi_cam_ds/calibration/footage/cam_1.mp4",
   "/home/piyansh/multi_cam_ds/calibration/footage/cam_2.mp4",
   "/home/piyansh/multi_cam_ds/calibration/footage/cam_3.mp4",
   "/home/piyansh/multi_cam_ds/calibration/footage/cam_4.mp4"
]

TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDeepSORT.yml"

# Settings
USE_ALTERNATIVE_SETTINGS = False   # Use alternative settings for contrast and threshold
DEFAULT_THRESHOLD = 0.25 
ALTERNATIVE_THRESHOLD = 0.15
DISPLAY_VIDEO = False
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
PERSON_CLASS_ID = 0
DEBUG_MODE = True

# Display colors for different cameras
COLORS = [
   (0.0, 0.5, 1.0, 0.5),  # Blue
   (0.0, 1.0, 0.0, 0.5),  # Green
   (1.0, 0.0, 0.0, 0.5),  # Red
   (1.0, 1.0, 0.0, 0.5),  # Yellow
]

# Buffers for centroids
centroid_buffers = {}
last_write_time = {}

def log_debug(msg):
    if DEBUG_MODE:
       print(f"DEBUG: {msg}")

# Flush buffers in background
def flush_buffers_periodically():
    while True:
        current_time = time.time()
        for source_id, buffer in list(centroid_buffers.items()):
            # Flush when buffer gets large or after timeout
            if buffer and (len(buffer) >= 30 or 
                          (source_id in last_write_time and 
                           current_time - last_write_time.get(source_id, 0) > 3)):
                file_name = f"centroids_source_{source_id}.csv"
                if USE_ALTERNATIVE_SETTINGS:
                    file_name = f"alt_centroids_source_{source_id}.csv"
                    
                try:
                    with open(file_name, "a") as f:
                        f.write("\n".join(buffer) + "\n")
                    centroid_buffers[source_id] = []
                    last_write_time[source_id] = current_time
                    if DEBUG_MODE:
                        print(f"Flushed {len(buffer)} centroids for source {source_id}")
                except Exception as e:
                    print(f"Error writing to {file_name}: {e}")
        time.sleep(1)

# Filter for people only
def pgie_src_pad_buffer_probe(pad, info, u_data):
    source_id = u_data
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK
 
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
 
    l_frame = batch_meta.frame_meta_list
    frame_count = 0
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_count += 1
        except StopIteration:
            break
 
        l_obj = frame_meta.obj_meta_list
        obj_count = 0
        person_count = 0
       
        # Filter out non-persons
        current_l_obj = l_obj
        while current_l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(current_l_obj.data)
                next_obj = current_l_obj.next
                obj_count += 1
 
                threshold = ALTERNATIVE_THRESHOLD if USE_ALTERNATIVE_SETTINGS else DEFAULT_THRESHOLD
                
                # Remove if not a person or below threshold
                if obj_meta.class_id != PERSON_CLASS_ID or obj_meta.confidence < threshold:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                else:
                    # Enhance person display
                    obj_meta.rect_params.border_width = 2
                    obj_meta.text_params.font_params.font_size = 14
                    obj_meta.text_params.set_bg_clr = 1
                    person_count += 1
 
                current_l_obj = next_obj
            except StopIteration:
                break
 
        log_debug(f"Source {source_id}, Frame {frame_meta.frame_num}: Found {person_count}/{obj_count} objects")
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
 
    log_debug(f"Processed {frame_count} frames in batch")
    return Gst.PadProbeReturn.OK

# Add source ID and save centroids
def source_pad_buffer_probe(pad, info, u_data):
    source_id = u_data
   
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK
 
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
 
    l_frame = batch_meta.frame_meta_list
    frame_count = 0
    
    # Init buffer for this source
    if source_id not in centroid_buffers:
        centroid_buffers[source_id] = []
        last_write_time[source_id] = time.time()
    
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_count += 1
            frame_meta.source_id = source_id
        except StopIteration:
            break
 
        l_obj = frame_meta.obj_meta_list
        obj_count = 0
       
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
               
                if obj_meta.class_id == PERSON_CLASS_ID:
                    # Style the detection
                    obj_meta.rect_params.border_width = 3
                    obj_meta.rect_params.has_bg_color = 1
                    color = COLORS[source_id % len(COLORS)]
                    obj_meta.rect_params.bg_color.set(*color)
                    obj_meta.text_params.font_params.font_size = 14
                    obj_meta.text_params.set_bg_clr = 1
                    obj_meta.text_params.display_text = f"Cam{source_id}-ID{obj_meta.object_id}"
                   
                    # Calculate centroid
                    rect = obj_meta.rect_params
                    cx = rect.left + rect.width / 2
                    cy = rect.top + rect.height / 2
                   
                    # Save centroid
                    centroid_buffers[source_id].append(f"{frame_meta.frame_num},{obj_meta.object_id},{cx},{cy}")
                    obj_count += 1
               
                l_obj = l_obj.next
            except StopIteration:
                break
 
        log_debug(f"Source {source_id}, Frame {frame_meta.frame_num}: Processed {obj_count} people")
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
 
    return Gst.PadProbeReturn.OK

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        if "No valid frames found before end of stream" in str(err):
            print(f"Invalid stream detected, but continuing with others")
        elif "Output window was closed" in str(err):
            print("Display window was closed, continuing processing without display")
        else:
            loop.quit()
    return True
 
def main():
    # Show current mode
    if USE_ALTERNATIVE_SETTINGS:
        print(f"Using ALTERNATIVE settings: contrast boost, threshold={ALTERNATIVE_THRESHOLD}")
    else:
        print(f"Using DEFAULT settings: normal contrast, threshold={DEFAULT_THRESHOLD}")
        
    # Init GStreamer
    Gst.init(None)
   
    # Check files
    for video_file in VIDEO_FILES:
        if not os.path.isfile(video_file):
            print(f"File not found: {video_file}")
            sys.exit(1)
       
    if not os.path.isfile(YOLO_CONFIG_PATH):
        print(f"YOLO config not found: {YOLO_CONFIG_PATH}")
        sys.exit(1)
       
    if not os.path.isfile(TRACKER_CONFIG):
        print(f"Tracker config not found: {TRACKER_CONFIG}")
        sys.exit(1)
   
    print("Creating pipeline with separate chains per source")
    pipeline = Gst.Pipeline()
   
    source_muxers = []
    source_converters = []
   
    # Final muxer for display
    final_mux = Gst.ElementFactory.make("nvstreammux", "final-muxer")
    final_mux.set_property("width", 1280)
    final_mux.set_property("height", 720)
    final_mux.set_property("batch-size", len(VIDEO_FILES))
    final_mux.set_property("batched-push-timeout", 40000)
    final_mux.set_property("attach-sys-ts", True)
    pipeline.add(final_mux)
   
    # Create chain for each source
    for i, video_file in enumerate(VIDEO_FILES):
        print(f"Creating chain for source {i}: {video_file}")
       
        # Video source elements
        source = Gst.ElementFactory.make("filesrc", f"file-source-{i}")
        source.set_property("location", video_file)
        pipeline.add(source)
       
        demux = Gst.ElementFactory.make("qtdemux", f"qtdemux-{i}")
        pipeline.add(demux)
       
        parser = Gst.ElementFactory.make("mpeg4videoparse", f"mpeg4-parser-{i}")
        pipeline.add(parser)
       
        decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{i}")
        decoder.set_property("drop-frame-interval", 0)
        decoder.set_property("num-extra-surfaces", 5)
        pipeline.add(decoder)
       
        # Source-specific muxer
        mux = Gst.ElementFactory.make("nvstreammux", f"stream-muxer-{i}")
        mux.set_property("width", 1280)
        mux.set_property("height", 720)
        mux.set_property("batch-size", 1)
        mux.set_property("batched-push-timeout", 40000)
        mux.set_property("attach-sys-ts", True)
        pipeline.add(mux)
        source_muxers.append(mux)
        
        # Optional contrast enhancement
        if USE_ALTERNATIVE_SETTINGS:
            pre_convert = Gst.ElementFactory.make("nvvideoconvert", f"pre-convert-{i}")
            pipeline.add(pre_convert)
            
            videobox = Gst.ElementFactory.make("videobox", f"videobox-{i}")
            if videobox:
                videobox.set_property("border-alpha", 0)
            pipeline.add(videobox)
            
            videobalance = Gst.ElementFactory.make("videobalance", f"videobalance-{i}")
            if videobalance:
                videobalance.set_property("contrast", 1.4)
                videobalance.set_property("brightness", 0.1)
            pipeline.add(videobalance)
            
            post_convert = Gst.ElementFactory.make("nvvideoconvert", f"post-convert-{i}")
            pipeline.add(post_convert)
       
        # Inference and tracking
        pgie = Gst.ElementFactory.make("nvinfer", f"primary-inference-{i}")
        pgie.set_property("config-file-path", YOLO_CONFIG_PATH)
        if os.path.isfile(YOLO_ENGINE_PATH):
            pgie.set_property("model-engine-file", YOLO_ENGINE_PATH)
        pgie.set_property("batch-size", 1)
        pgie.set_property("unique-id", 1 + i*10)
        pipeline.add(pgie)
       
        queue = Gst.ElementFactory.make("queue", f"queue-{i}")
        queue.set_property("max-size-buffers", 3)
        pipeline.add(queue)
       
        tracker = Gst.ElementFactory.make("nvtracker", f"tracker-{i}")
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so")
        tracker.set_property("ll-config-file", TRACKER_CONFIG)
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)
        tracker.set_property("gpu-id", 0)
        tracker.set_property("display-tracking-id", 1)
        pipeline.add(tracker)
       
        converter = Gst.ElementFactory.make("nvvideoconvert", f"converter-{i}")
        pipeline.add(converter)
        source_converters.append(converter)
       
        osd = Gst.ElementFactory.make("nvdsosd", f"onscreen-display-{i}")
        pipeline.add(osd)
       
        # Link base elements
        source.link(demux)
        parser.link(decoder)
       
        # Handle demuxer dynamic pads
        def on_pad_added(demuxer, pad, idx=i, parser_elem=parser):
            if not pad.get_name().startswith("video_"):
                print(f"Source {idx}: Not a video pad")
                return
               
            parser_sink = parser_elem.get_static_pad("sink")
            if parser_sink.is_linked():
                print(f"Source {idx}: Parser sink already linked")
                return
               
            if pad.link(parser_sink) != Gst.PadLinkReturn.OK:
                print(f"Source {idx}: Failed to link demuxer to parser")
            else:
                print(f"Source {idx}: Successfully linked demuxer to parser")
       
        demux.connect("pad-added", on_pad_added)
       
        # Get muxer input pad
        mux_sink = mux.get_request_pad(f"sink_0")
       
        # Link differently based on settings
        if USE_ALTERNATIVE_SETTINGS:
            # Link with contrast enhancement
            if decoder.get_static_pad("src"):
                decoder.link(pre_convert)
                pre_convert.link(videobox)
                videobox.link(videobalance)
                videobalance.link(post_convert)
                post_convert_src = post_convert.get_static_pad("src")
                if post_convert_src.link(mux_sink) != Gst.PadLinkReturn.OK:
                    print(f"Source {i}: Failed to link post-convert to muxer")
                else:
                    print(f"Source {i}: Successfully linked post-convert to muxer")
            else:
                # Try linking later
                def try_link_decoder():
                    if decoder.get_static_pad("src"):
                        decoder.link(pre_convert)
                        pre_convert.link(videobox)
                        videobox.link(videobalance)
                        videobalance.link(post_convert)
                        post_convert_src = post_convert.get_static_pad("src")
                        if post_convert_src.link(mux_sink) != Gst.PadLinkReturn.OK:
                            print(f"Source {i}: Failed to link post-convert to muxer")
                        else:
                            print(f"Source {i}: Successfully linked post-convert to muxer")
                        return False
                    return True
                GLib.timeout_add(500, try_link_decoder)
        else:
            # Standard linking
            if decoder.get_static_pad("src"):
                if decoder.get_static_pad("src").link(mux_sink) != Gst.PadLinkReturn.OK:
                    print(f"Source {i}: Failed to link decoder to muxer")
                else:
                    print(f"Source {i}: Successfully linked decoder to muxer")
            else:
                # Try linking later
                def try_link_decoder():
                    src_pad = decoder.get_static_pad("src")
                    if src_pad:
                        if src_pad.link(mux_sink) != Gst.PadLinkReturn.OK:
                            print(f"Source {i}: Failed to link decoder to muxer")
                        else:
                            print(f"Source {i}: Successfully linked decoder to muxer")
                        return False
                    return True
                GLib.timeout_add(500, try_link_decoder)
       
        # Link remaining elements
        mux.link(pgie)
        pgie.link(queue)
        queue.link(tracker)
        tracker.link(converter)
        converter.link(osd)
       
        # Add probes for filtering and metadata
        pgie_src_pad = pgie.get_static_pad("src")
        if pgie_src_pad:
            pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, i)
        else:
            print(f"WARNING: Could not add person filter probe for source {i}")
       
        tracker_src = tracker.get_static_pad("src")
        tracker_src.add_probe(Gst.PadProbeType.BUFFER, source_pad_buffer_probe, i)
       
        # Prepare output files
        file_prefix = "alt_" if USE_ALTERNATIVE_SETTINGS else ""
        with open(f"{file_prefix}centroids_source_{i}.csv", "w") as f:
            f.write("frame,person_id,centroid_x,centroid_y\n")
   
    # Connect each source's OSD to final muxer
    for i in range(len(VIDEO_FILES)):
        osd = pipeline.get_by_name(f"onscreen-display-{i}")
        osd_src = osd.get_static_pad("src")
        final_mux_sink = final_mux.get_request_pad(f"sink_{i}")
       
        if osd_src.link(final_mux_sink) != Gst.PadLinkReturn.OK:
            print(f"Failed to link OSD {i} to final muxer")
        else:
            print(f"Successfully linked OSD {i} to final muxer")
   
    # Complete display pipeline
    if DISPLAY_VIDEO:
        # Set up tiled display
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not tiler:
            print("Failed to create tiler")
            sys.exit(1)
       
        # Arrange videos in grid
        rows = int(len(VIDEO_FILES) ** 0.5)
        if rows ** 2 < len(VIDEO_FILES):
            rows += 1
        cols = (len(VIDEO_FILES) + rows - 1) // rows
       
        tiler.set_property("rows", rows)
        tiler.set_property("columns", cols)
        tiler.set_property("width", WINDOW_WIDTH)
        tiler.set_property("height", WINDOW_HEIGHT)
        pipeline.add(tiler)
       
        # Final display elements
        conv_final = Gst.ElementFactory.make("nvvideoconvert", "final-converter")
        pipeline.add(conv_final)
       
        # Try different display sinks
        sink = Gst.ElementFactory.make("ximagesink", "video-renderer")
        if not sink:
            print("Could not create ximagesink, trying nveglglessink")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                print("No display sink available, using fakesink")
                sink = Gst.ElementFactory.make("fakesink", "fakesink")
       
        pipeline.add(sink)
       
        # Link display
        final_mux.link(tiler)
        tiler.link(conv_final)
        conv_final.link(sink)
    else:
        # No display
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        pipeline.add(sink)
        final_mux.link(sink)
    
    # Start buffer thread
    buffer_thread = threading.Thread(target=flush_buffers_periodically, daemon=True)
    buffer_thread.start()
   
    # Run
    print(f"Starting pipeline (threshold: {ALTERNATIVE_THRESHOLD if USE_ALTERNATIVE_SETTINGS else DEFAULT_THRESHOLD})")
    pipeline.set_state(Gst.State.PLAYING)
   
    # Event loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
   
    try:
        print("Running pipeline...")
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        # Save any remaining data
        for source_id, buffer in centroid_buffers.items():
            if buffer:
                file_prefix = "alt_" if USE_ALTERNATIVE_SETTINGS else ""
                file_name = f"{file_prefix}centroids_source_{source_id}.csv"
                with open(file_name, "a") as f:
                    f.write("\n".join(buffer) + "\n")
        pipeline.set_state(Gst.State.NULL)
   
    print("Done")
 
if __name__ == "__main__":
    main()
 
