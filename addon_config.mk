meta:
	ADDON_NAME = ofxYoloPose
	ADDON_DESCRIPTION = People and their 17 COCO keypoints in one pass with Ultralytics YOLO11-pose (ONNX), tracked from frame to frame
	ADDON_AUTHOR = danm
	ADDON_TAGS = "yolo" "yolo11" "pose" "onnx" "machine learning" "inference" "pose tracker"
	ADDON_URL = https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX

common:
	ADDON_DEFINES = OF_ADDON_HAS_OFX_YOLO_POSE
	ADDON_DEPENDENCIES = ofxOnnx
	ADDON_DEPENDENCIES += ofxOpenCv
