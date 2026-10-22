#include "ofxYoloPose.h"

//--------------------------------------------------------------
ofxYoloPose::ofxYoloPose() {
    params.setName( "YOLO pose" );
    params.add( minScore, nmsIou, trackIou, forgetSec );
    params.add( smooth, minCutoff, beta );
}

//--------------------------------------------------------------
bool ofxYoloPose::setup( const Settings& aSettings ) {
    ofxOnnx::Settings os;
    os.modelPath = aSettings.modelPath;
    os.envName = "YoloPose";
    os.useCoreML = aSettings.useCoreML;
    os.logLevel = aSettings.logLevel;
    if( !mOnnx.load( os )) {
        ofLogError( "ofxYoloPose" ) << "could not load " << aSettings.modelPath;
        return false;
    }
    // the input size from the model: [1, 3, H, W]
    if( mOnnx.getInputCount() > 0 ) {
        const auto tshape = mOnnx.getInputShape( 0 );
        if( tshape.size() == 4 && tshape[2] > 0 && tshape[3] > 0 ) {
            mInH = (int)tshape[2];
            mInW = (int)tshape[3];
        }
    }
    mInput.assign( (size_t)3 * mInW * mInH, 0.f );
    mVerbose = aSettings.logLevel <= OF_LOG_VERBOSE;
    mLoaded = true;
    ofLogNotice( "ofxYoloPose" ) << "loaded " << aSettings.modelPath.filename() << ", input " << mInW << "x" << mInH;
    return true;
}

//--------------------------------------------------------------
bool ofxYoloPose::run( const ofPixels& aRgb ) {
    if( !isSetup() || !aRgb.isAllocated() || aRgb.getNumChannels() < 3 ) return false;

    // letterbox into the model's input: scaled to fit, centred, the rest the grey Ultralytics pads with
    const int tw = (int)aRgb.getWidth(), th = (int)aRgb.getHeight();
    const float tscale = std::min( (float)mInW / tw, (float)mInH / th );
    const int nw = std::max( 1, (int)std::round( tw * tscale )), nh = std::max( 1, (int)std::round( th * tscale ));
    const glm::vec2 tpad(( mInW - nw ) * 0.5f, ( mInH - nh ) * 0.5f );
    const cv::Mat tsrc( th, tw, aRgb.getNumChannels() == 4 ? CV_8UC4 : CV_8UC3, (void*)aRgb.getData(), aRgb.getBytesStride() );
    cv::Mat trgb = tsrc;
    if( aRgb.getNumChannels() == 4 ) cv::cvtColor( tsrc, trgb, cv::COLOR_RGBA2RGB );
    cv::resize( trgb, mResized, cv::Size( nw, nh ), 0, 0, cv::INTER_LINEAR );
    mPadded.create( mInH, mInW, CV_8UC3 );
    mPadded.setTo( cv::Scalar( 114, 114, 114 ));
    const int px = (int)std::floor( tpad.x ), py = (int)std::floor( tpad.y );
    mResized.copyTo( mPadded( cv::Rect( px, py, nw, nh )));

    // HWC bytes to CHW floats, 0..1
    const size_t tplane = (size_t)mInW * mInH;
    const uint8_t* tp = mPadded.data;
    float* r = mInput.data();
    float* g = r + tplane;
    float* b = g + tplane;
    for( size_t i = 0; i < tplane; i++ ) {
        r[i] = tp[3 * i] * ( 1.f / 255.f );
        g[i] = tp[3 * i + 1] * ( 1.f / 255.f );
        b[i] = tp[3 * i + 2] * ( 1.f / 255.f );
    }

    std::vector< Ort::Value > touts;
    try {
        touts = mOnnx.run( mInput, std::vector< int64_t >{ 1, 3, mInH, mInW } );
    } catch( const Ort::Exception& e ) {
        ofLogError( "ofxYoloPose" ) << "run: " << e.what();
        return false;
    }
    if( touts.empty() || !touts[0].IsTensor() ) return false;
    const auto tshape = touts[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector< Detection > tdets = decode( touts[0].GetTensorData< float >(), tshape, tscale, glm::vec2( (float)px, (float)py ));

    if( mVerbose && ofGetElapsedTimeMicros() - mLastVerboseUs > 2000000 ) {
        mLastVerboseUs = ofGetElapsedTimeMicros();
        std::string ts;
        for( auto v : tshape ) ts += ofToString( v ) + " ";
        // the best candidate's score, whether or not it passed, and the raw value ranges
        const float* to = touts[0].GetTensorData< float >();
        const size_t tcount = touts[0].GetTensorTypeAndShapeInfo().GetElementCount();
        float tmin = 1e9f, tmax = -1e9f;
        for( size_t i = 0; i < tcount; i++ ) { tmin = std::min( tmin, to[i] ); tmax = std::max( tmax, to[i] ); }
        ofLogNotice( "ofxYoloPose" ) << "output [ " << ts << "], values " << tmin << " .. " << tmax << ", picture " << tw << "x" << th
                                     << ", scale " << tscale << ", pad " << px << "," << py << ", people " << tdets.size();
        if( !tdets.empty() ) {
            const Detection& d = tdets[0];
            std::string tk;
            for( int k : { 0, 5, 6, 9, 10, 11, 12 } ) tk += ofToString( k ) + ":(" + ofToString( d.keypoints[k].x, 0 ) + "," + ofToString( d.keypoints[k].y, 0 ) + " " + ofToString( d.scores[k], 2 ) + ") ";
            ofLogNotice( "ofxYoloPose" ) << "best: score " << d.score << ", box " << d.box << ", keypoints " << tk;
        }
    }

    // the clock, for forgetting people and the rate
    const uint64_t tnow = ofGetElapsedTimeMicros();
    const float tdt = mLastRunUs ? ( tnow - mLastRunUs ) * 1e-6f : 0.f;
    mLastRunUs = tnow;
    if( tnow - mSecondStartUs >= 1000000 ) {
        mFps = mRunsThisSecond * 1e6f / std::max< uint64_t >( 1, tnow - mSecondStartUs );
        mRunsThisSecond = 0;
        mSecondStartUs = tnow;
    }
    mRunsThisSecond++;

    follow( tdets, tdt );
    return true;
}

//--------------------------------------------------------------
// The model's output, [1, 56, N] (or [1, N, 56]): for each of N candidates the box (centre x, y, width, height),
// the person score, and x, y and visibility for each keypoint, in the model's input pixels. Those over the score,
// with overlapping ones merged (NMS), back in the picture's own pixels.
std::vector< ofxYoloPose::Detection > ofxYoloPose::decode( const float* aOut, const std::vector< int64_t >& aShape, float aScale, glm::vec2 aPad ) const {
    std::vector< Detection > tcands;
    if( aShape.size() != 3 ) return tcands;
    const int tvals = 5 + 3 * kNumKeypoints;   // 56
    bool tchannelsFirst;
    int tn;
    if( aShape[1] == tvals ) { tchannelsFirst = true; tn = (int)aShape[2]; }
    else if( aShape[2] == tvals ) { tchannelsFirst = false; tn = (int)aShape[1]; }
    else {
        ofLogError( "ofxYoloPose" ) << "unexpected output shape [" << aShape[0] << ", " << aShape[1] << ", " << aShape[2] << "]";
        return tcands;
    }
    auto at = [&]( int aValue, int aCand ) { return tchannelsFirst ? aOut[(size_t)aValue * tn + aCand] : aOut[(size_t)aCand * tvals + aValue]; };
    auto back = [&]( float x, float y ) { return glm::vec2(( x - aPad.x ) / aScale, ( y - aPad.y ) / aScale ); };

    const float tmin = minScore;
    for( int i = 0; i < tn; i++ ) {
        const float tscore = at( 4, i );
        if( tscore < tmin ) continue;
        Detection d;
        d.score = tscore;
        const float cx = at( 0, i ), cy = at( 1, i ), w = at( 2, i ), h = at( 3, i );
        const glm::vec2 a = back( cx - w * 0.5f, cy - h * 0.5f ), b = back( cx + w * 0.5f, cy + h * 0.5f );
        d.box.set( a.x, a.y, b.x - a.x, b.y - a.y );
        for( int k = 0; k < kNumKeypoints; k++ ) {
            d.keypoints[k] = back( at( 5 + 3 * k, i ), at( 6 + 3 * k, i ));
            d.scores[k] = at( 7 + 3 * k, i );
        }
        tcands.push_back( d );
    }

    // NMS: the best first, dropping those that overlap one kept
    std::sort( tcands.begin(), tcands.end(), []( const Detection& a, const Detection& b ) { return a.score > b.score; } );
    std::vector< Detection > tkept;
    const float tiou = nmsIou;
    for( auto& c : tcands ) {
        bool tdup = false;
        for( auto& k : tkept ) if( iou( c.box, k.box ) > tiou ) { tdup = true; break; }
        if( !tdup ) tkept.push_back( c );
    }
    return tkept;
}

//--------------------------------------------------------------
// The people found now matched to those being kept, best overlap first; the rest are new people. Someone not
// found is kept, as not found, for forgetSec.
void ofxYoloPose::follow( std::vector< Detection >& aDets, float aDt ) {
    struct Pair { float iou; int det, pose; };
    std::vector< Pair > tpairs;
    for( int d = 0; d < (int)aDets.size(); d++ )
        for( int p = 0; p < (int)mPoses.size(); p++ ) {
            const float t = iou( aDets[d].box, mPoses[p].box );
            if( t >= trackIou ) tpairs.push_back( { t, d, p } );
        }
    std::sort( tpairs.begin(), tpairs.end(), []( const Pair& a, const Pair& b ) { return a.iou > b.iou; } );
    std::vector< int > tdetPose( aDets.size(), -1 );
    std::vector< bool > tposeUsed( mPoses.size(), false );
    for( auto& pr : tpairs ) {
        if( tdetPose[pr.det] >= 0 || tposeUsed[pr.pose] ) continue;
        tdetPose[pr.det] = pr.pose;
        tposeUsed[pr.pose] = true;
    }

    for( auto& p : mPoses ) {
        p.foundThisFrame = false;
        p.timeNotFound += aDt;
    }
    for( int d = 0; d < (int)aDets.size(); d++ ) {
        int tp = tdetPose[d];
        if( tp < 0 ) {
            Pose tnew;
            tnew.id = mNextId++;
            mPoses.push_back( tnew );
            tp = (int)mPoses.size() - 1;
        }
        Pose& p = mPoses[tp];
        p.score = aDets[d].score;
        p.scores = aDets[d].scores;
        if( smooth && p.filtered && aDt > 0.f ) {
            // someone already followed: their keypoints and box through the filters
            for( int k = 0; k < kNumKeypoints; k++ ) p.keypoints[k] = oneEuro( aDets[d].keypoints[k], p.fx[k], p.fdx[k], aDt );
            const glm::vec2 ta = oneEuro( glm::vec2( aDets[d].box.x, aDets[d].box.y ), p.fbox[0], p.fdbox[0], aDt );
            const glm::vec2 tb = oneEuro( glm::vec2( aDets[d].box.getRight(), aDets[d].box.getBottom() ), p.fbox[1], p.fdbox[1], aDt );
            p.box.set( ta.x, ta.y, tb.x - ta.x, tb.y - ta.y );
        } else {
            // someone new (or smoothing off): as found, and the filters start from here
            p.keypoints = aDets[d].keypoints;
            p.box = aDets[d].box;
            for( int k = 0; k < kNumKeypoints; k++ ) { p.fx[k] = p.keypoints[k]; p.fdx[k] = glm::vec2( 0.f ); }
            p.fbox = { glm::vec2( p.box.x, p.box.y ), glm::vec2( p.box.getRight(), p.box.getBottom() ) };
            p.fdbox = { glm::vec2( 0.f ), glm::vec2( 0.f ) };
            p.filtered = true;
        }
        p.foundThisFrame = true;
        p.timeNotFound = 0.f;
    }
    const float tforget = forgetSec;
    mPoses.erase( std::remove_if( mPoses.begin(), mPoses.end(), [&]( const Pose& p ) { return !p.foundThisFrame && p.timeNotFound > tforget; } ), mPoses.end() );
}

//--------------------------------------------------------------
// One Euro filter (Casiez, Roussel, Vogel 2012): a low-pass whose cutoff rises with the speed, so a still joint is
// held steady and a moving one is followed with little lag.
glm::vec2 ofxYoloPose::oneEuro( glm::vec2 aX, glm::vec2& aPrev, glm::vec2& aPrevDx, float aDt ) const {
    auto alpha = [&]( float aCutoff ) {
        const float tau = 1.f / ( glm::two_pi< float >() * aCutoff );
        return 1.f / ( 1.f + tau / aDt );
    };
    const glm::vec2 tdx = ( aX - aPrev ) / aDt;
    aPrevDx = glm::mix( aPrevDx, tdx, alpha( 1.f ));            // the speed itself smoothed, at 1 Hz
    const float tcut = minCutoff + beta * glm::length( aPrevDx );
    aPrev = glm::mix( aPrev, aX, alpha( tcut ));
    return aPrev;
}

//--------------------------------------------------------------
float ofxYoloPose::iou( const ofRectangle& a, const ofRectangle& b ) {
    const ofRectangle i = a.getIntersection( b );
    const float ti = i.getArea();
    const float tu = a.getArea() + b.getArea() - ti;
    return tu > 0.f ? ti / tu : 0.f;
}
