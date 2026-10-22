# ofxYoloPose

People and their 17 COCO keypoints in one pass, with Ultralytics **YOLO11-pose** run through ONNX Runtime ([ofxOnnx](https://github.com/danzeeeman/ofxOnnx)), and kept from frame to frame with an id each.

Made for [femtoSkeletonOsc](https://github.com/danzeeeman/EchoAndNarcissus), which puts the keypoints in 3D with an Orbbec Femto's depth image, but it only needs a picture.

## Models

The ONNX models are exported from Ultralytics' `yolo11*-pose.pt` and hosted on Hugging Face:

**https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX**

| model | size | |
|---|---|---|
| [yolo11n-pose.onnx](https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/main/yolo11n-pose.onnx) | 12 MB | fastest; about 30 runs a second on 1280x720 on an Apple silicon Mac |
| [yolo11s-pose.onnx](https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/main/yolo11s-pose.onnx) | 40 MB | |
| [yolo11m-pose.onnx](https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/main/yolo11m-pose.onnx) | 84 MB | |
| [yolo11l-pose.onnx](https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/main/yolo11l-pose.onnx) | 105 MB | |
| [yolo11x-pose.onnx](https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/main/yolo11x-pose.onnx) | 235 MB | most accurate |

Put one in your app's `bin/data` (the models are not in this repo), for example:

```sh
mkdir -p bin/data/models/YoloPose
curl -L -o bin/data/models/YoloPose/yolo11n-pose.onnx \
  https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX/resolve/main/yolo11n-pose.onnx
```

Any YOLO11 (or YOLOv8) pose model exported to ONNX the Ultralytics way works: input `images` `[1, 3, H, W]` (the size is read from the model), output `[1, 56, N]` or `[1, N, 56]`.

YOLO11 and its weights are © Ultralytics and licensed **AGPL-3.0** (or under an Ultralytics enterprise license); check that this fits how you use them.

## Usage

`addons.make`:

```
ofxOnnx
ofxOpenCv
ofxYoloPose
```

```cpp
#include "ofxYoloPose.h"

ofxYoloPose yolo;

// setup()
ofxYoloPose::Settings s;
s.modelPath = ofToDataPath( "models/YoloPose/yolo11n-pose.onnx", true );
yolo.setup( s );

// each new frame (RGB ofPixels)
if( yolo.run( pixels )) {
    for( auto& p : yolo.getPoses() ) {
        if( !p.foundThisFrame ) continue;
        // p.id, p.box (pixels), p.score
        // p.keypoints[k] (pixels) and p.scores[k] (0..1), k in COCO order:
        //   0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists,
        //   11/12 hips, 13/14 knees, 15/16 ankles (left, then right)
    }
}
```

`run()` is synchronous: it runs the model on the picture it is given and returns when the poses are ready. Call it from whichever thread owns the frames (a camera's capture thread, say) and read `getPoses()` on that thread. The poses always belong to the picture that was passed in, so anything taken with it, like a depth image, lines up.

## What it does

1. **Letterbox.** The picture is scaled to fit the model's input (640x640 for the hosted models), centred, and padded with Ultralytics' grey (114). It is converted to CHW floats, RGB, 0..1.
2. **Decode.** Each of the model's candidates (8400 at 640x640) has a box, a person score, and x, y and visibility for the 17 keypoints. Those over the minimum score are kept. Overlapping ones are merged (NMS), and everything is mapped back to the picture's own pixels.
3. **Follow.** People are matched to those already being kept by box overlap, best first. Someone new gets a new id. Someone not found is kept, as not found, for a moment before being forgotten.
4. **Smooth.** A [One Euro filter](https://gery.casiez.net/1euro/) runs on every keypoint and on the box. It holds a still joint steady and follows quick moves with little lag.

On macOS ofxOnnx runs the model with CoreML.

## Settings

`params` (an `ofParameterGroup`, for a GUI):

| | default | |
|---|---|---|
| Min person score | 0.4 | a candidate below this is not a person |
| Overlap to merge (IoU) | 0.5 | NMS: two boxes overlapping more than this are one person |
| Overlap to follow (IoU) | 0.3 | a person found now continues one kept if their boxes overlap this much |
| Forget a person after (s) | 0.5 | how long someone not found is kept |
| Smooth | on | the One Euro filter |
| Smooth: min cutoff (Hz) | 1.5 | lower: steadier when still |
| Smooth: speed response | 0.01 | higher: less lag on fast moves |

`Settings::logLevel = OF_LOG_VERBOSE` also logs what it decodes every couple of seconds (the output shape, the value range, the best person's box and keypoints), for checking a new model.

## Requires

- [ofxOnnx](https://github.com/danzeeeman/ofxOnnx) (ONNX Runtime)
- ofxOpenCv (ships with openFrameworks)
