# YOLO-Based Social Costmap for TIAGo Robot

This implementation provides real-time human detection and social costmap generation for the TIAGo robot using YOLOv11n segmentation.

## Architecture

The system is modular with three core components:

### 1. **PersonDetector** (`person_detector.py`)
- Loads YOLOv11n-seg model for person detection
- Runs inference on RGB images at 5Hz (configurable)
- Filters for "person" class detections
- Returns bounding boxes and segmentation masks

### 2. **PersonLocalizer** (`person_localizer.py`)
- Converts detected persons to 3D positions using depth data
- Two localization methods:
  - **3d_centroid** (default): Median of valid depth samples within segmentation mask
  - **min_z_column**: Minimum depth per column with x-axis mirroring for precise edges
- Transforms positions from camera frame to map frame using TF2

### 3. **SocialCostmapPublisher** (`social_costmap_publisher.py`)
- Generates `OccupancyGrid` with three-zone social cost model:
  - **Intimate zone** (0-0.5m): 100% cost
  - **Personal zone** (0.5-1.2m): 70-100% cost (Gaussian falloff)
  - **Social zone** (1.2-3.6m): 30-70% cost (Gaussian falloff)
- Publishes to `/social_costmap` topic
- Publishes visualization markers to `/social_costmap/person_markers`

### 4. **SocialCostmapNode** (`social_costmap_node.py`)
- Orchestrates the full pipeline
- Synchronizes RGB-D camera streams
- Rate-limits detection to 5Hz
- Publishes social costmap and markers

## Configuration

### ROS Parameters

```yaml
detection_rate: 5.0                    # Hz (YOLO inference rate)
yolo_model: 'yolo11n-seg.pt'          # Model variant
confidence_threshold: 0.5              # Detection confidence
device: 'cuda'                         # 'cuda' or 'cpu'
localization_method: '3d_centroid'    # or 'min_z_column'
costmap_resolution: 0.05               # meters per cell
costmap_width: 10.0                    # meters
costmap_height: 10.0                   # meters
intimate_radius: 0.5                   # meters
personal_radius: 1.2                   # meters
social_radius: 3.6                     # meters
save_debug_images: false               # Save annotated images
debug_dir: '~/src/tmp'                # Debug output directory
```

## Topics

**Subscribed:**
- `/head_front_camera/color/image_raw` - RGB camera
- `/head_front_camera/depth/image_raw` - Depth camera
- `/head_front_camera/color/camera_info` - RGB camera info
- `/head_front_camera/depth/camera_info` - Depth camera info

**Published:**
- `/social_costmap` - OccupancyGrid (nav_msgs/OccupancyGrid)
- `/social_costmap/person_markers` - Visualization markers (visualization_msgs/MarkerArray)

## Installation

Dependencies are automatically installed via Docker. The Dockerfile includes:
- `ultralytics>=8.1.0` - YOLOv11 framework
- `torch>=2.0.0` - PyTorch
- `torchvision>=0.15.0` - Computer vision utilities

## Usage

The node starts automatically with the simulation:

```bash
ros2 launch tiago_social_sim simulation.launch.py navigation:=True slam:=True
```

### Manual Launch

```bash
ros2 run tiago_social_nav social_costmap_node
```

### With Custom Parameters

```bash
ros2 run tiago_social_nav social_costmap_node --ros-args \
  -p detection_rate:=10.0 \
  -p confidence_threshold:=0.6 \
  -p localization_method:=min_z_column \
  -p save_debug_images:=true
```

## Visualization

In RViz2:
1. Add **Map** display for `/social_costmap`
2. Add **MarkerArray** for `/social_costmap/person_markers`
3. Green spheres show detected person positions
4. Costmap shows social zones around each person

## Nav2 Integration

The social costmap is integrated into Nav2's local costmap via `99_user_social_costmap.yaml`:
- Merges with existing obstacle layers
- Uses maximum cost combination method
- Influences path planning to avoid social zones

## Performance

- **Detection Rate**: 5Hz (200ms intervals)
- **Model**: YOLOv11n-seg (~50-100 FPS on GPU)
- **Latency**: ~20-30ms per detection + localization
- **Memory**: ~500MB GPU (YOLO model + depth processing)

## Debug Mode

Enable debug mode to save annotated images:

```bash
ros2 run tiago_social_nav social_costmap_node --ros-args \
  -p save_debug_images:=true \
  -p debug_dir:=/home/user/debug
```

Saved files:
- `detected_persons.jpg` - RGB with bounding boxes and masks
- `registered_depth.png` - Depth aligned to RGB (colormap)
- `persons.txt` - Person positions and metadata

## Future Enhancements

1. **Person Tracking**: Kalman filter for temporal consistency
2. **Directional Costs**: Asymmetric zones based on person orientation
3. **Velocity-Aware Zones**: Dynamic zone sizing based on movement
4. **Multi-Camera Fusion**: Combine detections from multiple cameras
5. **Custom Nav2 Plugin**: Native C++ costmap layer for lower latency
