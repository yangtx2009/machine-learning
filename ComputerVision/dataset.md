# Dataset
## COCO Detection 2014, 2015, 2017
- References
  - [Create COCO Annotations From Scratch](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format)
- Tasks
  - Captions
  - Segmentation
  - Keypoints
- API
  - https://github.com/cocodataset/cocoapi
  - Visualization tool: [FiftyOne](https://fiftyone.ai/)
- Format: JSON
```json
{
    "info": {
        "description": "COCO 2017 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2017,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
        },
    "licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        ...
    ],
    "images": [
        {
            "license": 4,
            "file_name": "000000397133.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "height": 427,
            "width": 640,
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
            "id": 397133
        },
        ...
    ],
    "annotations": [
        {
            "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
            "area": 702.1057499999998,
            "iscrowd": 0,
            "keypoints": [229,256,2,...,223,369,2],
            "image_id": 289343,
            "bbox": [473.07,395.93,38.65,28.67],
            "category_id": 18,
            "id": 1768
        },
        ...
    ],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ],
            "skeleton": [
                [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
            ] --> 19 keypoints
        },
        ...
    ], <-- Not in Captions annotations
    "segment_info": [...] <-- Only in Panoptic annotations
}
```

## KITTI
- References
- Tasks
  - 2D Object Detection/Segmentation
  - 3D Object Detection/Segmentation
  - Bird's Eye View (transformed by [development kit](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip))
- API
- Format:
  - [**camera calibration matrices**](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) --> timestamps.txt [calib] (e.g. 000000.txt)
    - P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo

  - **object data set** ([2D](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) & [3D](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)) --> timestamps.txt [label] (e.g. 000000.txt)
    - object type
    - truncated: 0~1
    - occluded Integer (0,1,2,3): 0 = fully visible,1 = partly occluded 2 = largely occluded, 3 = unknown
    - angle of object, ranging [-pi..pi]
    - bbox 2D bounding box: left, top, right, bottom pixel coordinates
    - 3D object dimensions: height, width, length (in meters)
    - 3D object location x,y,z in camera coordinates (in meters)
    - Rotation ry around Y-axis in camera coordinates [-pi..pi]
    - confidence in detection, needed for p/r curves, higher is better
  ```
  Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
  ```
  - [**semantic segmentation**](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)
    - follows format and metrics of Cityscapes Dataset
  - [**Velodyne point clouds**](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
    - bin (Velodyne fused point clouds BINARY format)
  

## [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/download.php)
- Tasks
  - 2D Object Detection/Segmentation
  - 3D Object Detection/Segmentation
- Format:
  - **data_2d_raw**: png, timestamps.txt (e.g. 000000.txt)
  - **data_2d_semantics**: png, instanceDict.json
  - **data_3d_raw**: bin (Velodyne fused point clouds BINARY format), timestamps.txt (e.g. 000000.txt)
  - **data_3d_semantics**: ply
  - **data_3d_bboxes**: xml
  - **data_poses**: txt


## [Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/)
- Tasks
  - Pixel-Level Semantic Labeling Task
  - Instance-Level Semantic Labeling Task
  - Panoptic Semantic Labeling Task
  - 3D Vehicle Detection Task
- Class definitions: [link](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions)
- Format:
  - Polygonal annotations
  - color segmentation (PNG)
  - instanceIds segmentation (PNG)
    - 16bit png label
    - stuff: 0~24
    - things: class ID × 1000 + X (e.g. 26001)
  - labelIds segmentation (PNG)
  - polygons (JSON)
  ```json
  {
    "imgHeight": 1024, 
    "imgWidth": 2048, 
    "objects": [
      {
        "label": "road", 
        "polygon": [ [0, 769], [290, 574], ...] // n polygon coordinates
      },
      // ... n objects in this image
    ]
  }
  ```

## [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- References
  - [The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Development Kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)
- Tasks
  - Classification/Detection
  - Action Classification
  - Segmentation: pixel-wise
  - Person Layout
- No 3D information
- API: 
  - [TensorFlow](https://www.tensorflow.org/datasets/catalog/voc)
  - [PyTorch](https://pytorch.org/vision/stable/_modules/torchvision/datasets/voc.html)
- Class definitions: [link](http://host.robots.ox.ac.uk/pascal/VOC/)
```xml
<annotation>
	<folder>VOC2007</folder>
	<filename>000005.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>325991873</flickrid>
	</source>
	<owner>
		<flickrid>archintent louisville</flickrid>
		<name>?</name>
	</owner>
	<size>
		<width>500</width>
		<height>375</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>chair</name>
		<pose>Rear</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>263</xmin>
			<ymin>211</ymin>
			<xmax>324</xmax>
			<ymax>339</ymax>
		</bndbox>
	</object>
    <object> ... </object>
</annotation>
```
- segmentation
  - 0: background
  - [1 .. 20] interval: segmented objects, classes [Aeroplane, ..., Tvmonitor]
  - 255: void category, used for border regions (5px) and to mask difficult objects

## [Waymo Open Dataset](https://waymo.com/open)
- References
  - [Label specifications document](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md)
- API
  - [Official Tutorial](https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb)
  - [TensorFlow](https://www.tensorflow.org/datasets/catalog/waymo_open_dataset)
  - [PyTorch](https://github.com/Manojbhat09/Waymo-pytorch-dataloader)
- Format
    ```
    center_x
    center_y
    center_z
    length
    width
    height
    heading
    speed_x
    speed_y
    accel_x
    accel_y
    ```
- [Labeling Specifications](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md)
  - Vehicle
    - Motorcycles and motorcyclists are labeled as vehicles
    - If a vehicle has any open doors (such as side doors, back doors or the gas tank lid), those are excluded from the bounding box
    - ...
  - Pedestrian
    - A single bounding box
    - If the pedestrian is carrying an object larger than 2m, or pushing a bike or shopping cart, the bounding box does not include the additional object.
    - If the pedestrian is pushing a stroller with a child in it, separate bounding boxes are created for the pedestrian and the child. The stroller is not included in the child bounding box.
    - Mannequins, statues, dummies, objects covered with cloth covers, billboards, posters, pedestrians inside buildings, or reflections of people are not labeled.
    - If pedestrians overlap each other, they are labeled as separate objects. If they overlap then the bounding boxes can overlap as well.
    - A person riding a bicycle is not labeled as a pedestrian, but labeled as a cyclist instead.
    - ...
  - Cyclist
    - ...
  - **Sign**
    - stop signs, yield signs, speed limit signs, warning signs, guide and recreation signs