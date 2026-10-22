// ofxYoloPose.h — people and their 17 COCO keypoints with Ultralytics YOLO11-pose (ONNX), in one pass
#pragma once
#include "ofMain.h"
#include "ofxOnnx.h"
#include "ofxOpenCv.h"
#include <array>
#include <memory>
#include <vector>

// Runs a YOLO11-pose model (yolo11n/s/m/l/x-pose.onnx, as Ultralytics exports it: input "images" [1, 3, 640, 640],
// RGB 0..1, output [1, 56, N]: box centre and size, the person score, then x, y and visibility for each of the
// 17 COCO keypoints) on a picture, and keeps the people from one picture to the next with an id each.
//
// run() is synchronous: call it from whichever thread owns the frames, and read getPoses() on that thread. That
// way the people always belong to the picture that was given (a depth image taken with it lines up).
//
// The models: https://huggingface.co/MikeLud/ObjectDetectionYOLO11-ONNX (yolo11n-pose.onnx is 12 MB).
class ofxYoloPose {
public:
    // COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
    static constexpr int kNumKeypoints = 17;

    struct Settings {
        of::filesystem::path modelPath;
        bool useCoreML = true;          // macOS
        ofLogLevel logLevel = OF_LOG_NOTICE;   // OF_LOG_VERBOSE: also what it decodes, every couple of seconds
    };

    // One person: where they are in the picture given to run() (pixels), the keypoints and how sure each is.
    struct Pose {
        uint64_t id = 0;
        ofRectangle box;
        float score = 0.f;
        std::array< glm::vec2, kNumKeypoints > keypoints;
        std::array< float, kNumKeypoints > scores{};
        bool foundThisFrame = false;
        float timeNotFound = 0.f;       // seconds since it was last found
        // the One Euro filters' state: last smoothed value and speed, per keypoint (x, y) and box corner
        std::array< glm::vec2, kNumKeypoints > fx, fdx;
        std::array< glm::vec2, 2 > fbox, fdbox;
        bool filtered = false;
    };

    ofxYoloPose();

    bool setup( const Settings& aSettings );
    bool isSetup() const { return mLoaded; }

    // Finds the people in an RGB picture. False when nothing could be run.
    bool run( const ofPixels& aRgb );
    // Everyone being kept: those found in the last picture (foundThisFrame) and, for a moment, those just lost.
    const std::vector< Pose >& getPoses() const { return mPoses; }

    // runs per second, over the last second
    float getFps() const { return mFps; }

    ofParameterGroup params;
    ofParameter< float > minScore{ "Min person score", 0.4f, 0.05f, 1.f };
    ofParameter< float > nmsIou{ "Overlap to merge (IoU)", 0.5f, 0.1f, 0.95f };
    ofParameter< float > trackIou{ "Overlap to follow (IoU)", 0.3f, 0.05f, 0.95f };
    ofParameter< float > forgetSec{ "Forget a person after (s)", 0.5f, 0.f, 5.f };
    // One Euro filter on the keypoints and the box: smooths hard while still, follows quick moves closely
    ofParameter< bool > smooth{ "Smooth", true };
    ofParameter< float > minCutoff{ "Smooth: min cutoff (Hz)", 1.5f, 0.05f, 10.f };
    ofParameter< float > beta{ "Smooth: speed response", 0.01f, 0.f, 0.2f };

protected:
    struct Detection {
        ofRectangle box;
        float score = 0.f;
        std::array< glm::vec2, kNumKeypoints > keypoints;
        std::array< float, kNumKeypoints > scores{};
    };
    std::vector< Detection > decode( const float* aOut, const std::vector< int64_t >& aShape, float aScale, glm::vec2 aPad ) const;
    void follow( std::vector< Detection >& aDets, float aDt );
    static float iou( const ofRectangle& a, const ofRectangle& b );
    // one step of a One Euro filter on a 2D point (pixels): `aX` the new value, `aPrev`/`aPrevDx` its state
    glm::vec2 oneEuro( glm::vec2 aX, glm::vec2& aPrev, glm::vec2& aPrevDx, float aDt ) const;

    ofxOnnx mOnnx;
    bool mLoaded = false;
    bool mVerbose = false;
    uint64_t mLastVerboseUs = 0;
    int mInW = 640, mInH = 640;
    std::vector< float > mInput;            // CHW, RGB, 0..1
    cv::Mat mResized, mPadded;

    std::vector< Pose > mPoses;
    uint64_t mNextId = 1;
    uint64_t mLastRunUs = 0;
    float mFps = 0.f;
    int mRunsThisSecond = 0;
    uint64_t mSecondStartUs = 0;
};
