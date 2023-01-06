//
// Created by star on 23-1-6.
//

#ifndef KALMAN_HOGDESC_H
#define KALMAN_HOGDESC_H

#include "opencv2/core.hpp"

namespace cv {

    class CV_EXPORTS SimilarRects
            {
                    public:
                    SimilarRects(double _eps) : eps(_eps) {}
                    inline bool operator()(const Rect& r1, const Rect& r2) const
                    {
                        double delta = eps * ((std::min)(r1.width, r2.width) + (std::min)(r1.height, r2.height)) * 0.5;
                        return std::abs(r1.x - r2.x) <= delta &&
                               std::abs(r1.y - r2.y) <= delta &&
                               std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
                               std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
                    }
                    double eps;
            };

    void myclipObjects(Size sz, std::vector<Rect>& objects,
                       std::vector<int>* a, std::vector<double>* b);


    CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps = 0.2);
/** @overload */
    CV_EXPORTS_W void groupRectangles(CV_IN_OUT std::vector<Rect>& rectList, CV_OUT std::vector<int>& weights,
                                      int groupThreshold, double eps = 0.2);
/** @overload */
    CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold,
                                      double eps, std::vector<int>* weights, std::vector<double>* levelWeights );
/** @overload */
    CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels,
                                      std::vector<double>& levelWeights, int groupThreshold, double eps = 0.2);
/** @overload */
    CV_EXPORTS   void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
                                                std::vector<double>& foundScales,
                                                double detectThreshold = 0.0, Size winDetSize = Size(64, 128));

    struct DetectionROI
    {
        //! scale(size) of the bounding box
        double scale;
        //! set of requested locations to be evaluated
        std::vector<cv::Point> locations;
        //! vector that will contain confidence values for each location
        std::vector<double> confidences;
    };

    struct CV_EXPORTS_W HOGDescriptor
            {
                    public:
                    enum HistogramNormType { L2Hys = 0 //!< Default histogramNormType
                    };
                    enum { DEFAULT_NLEVELS = 64 //!< Default nlevels value.
                    };
                    enum DescriptorStorageFormat { DESCR_FORMAT_COL_BY_COL, DESCR_FORMAT_ROW_BY_ROW };

                    /**@brief Creates the HOG descriptor and detector with default params.

                    aqual to HOGDescriptor(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9 )
                    */
                    CV_WRAP HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),
                    cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
                    histogramNormType(HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
                    free_coef(-1.f), nlevels(HOGDescriptor::DEFAULT_NLEVELS), signedGradient(false)
                    {}

                    /** @overload
                    @param _winSize sets winSize with given value.
                    @param _blockSize sets blockSize with given value.
                    @param _blockStride sets blockStride with given value.
                    @param _cellSize sets cellSize with given value.
                    @param _nbins sets nbins with given value.
                    @param _derivAperture sets derivAperture with given value.
                    @param _winSigma sets winSigma with given value.
                    @param _histogramNormType sets histogramNormType with given value.
                    @param _L2HysThreshold sets L2HysThreshold with given value.
                    @param _gammaCorrection sets gammaCorrection with given value.
                    @param _nlevels sets nlevels with given value.
                    @param _signedGradient sets signedGradient with given value.
                    */
                    CV_WRAP HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
                    Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
                    HOGDescriptor::HistogramNormType _histogramNormType=HOGDescriptor::L2Hys,
                    double _L2HysThreshold=0.2, bool _gammaCorrection=false,
                    int _nlevels=HOGDescriptor::DEFAULT_NLEVELS, bool _signedGradient=false)
                    : winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
                    nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
                    histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
                    gammaCorrection(_gammaCorrection), free_coef(-1.f), nlevels(_nlevels), signedGradient(_signedGradient)
                    {}

                    /** @overload
                    @param filename The file name containing HOGDescriptor properties and coefficients for the linear SVM classifier.
                    */
                    CV_WRAP HOGDescriptor(const String& filename)
                    {
                        load(filename);
                    }

                    /** @overload
                    @param d the HOGDescriptor which cloned to create a new one.
                    */
                    HOGDescriptor(const HOGDescriptor& d)
                    {
                        d.copyTo(*this);
                    }

                    /**@brief Default destructor.
                    */
                    virtual ~HOGDescriptor() {}

                    /**@brief Returns the number of coefficients required for the classification.
                    */
                    CV_WRAP size_t getDescriptorSize() const;

                    /** @brief Checks if detector size equal to descriptor size.
                    */
                    CV_WRAP bool checkDetectorSize() const;

                    /** @brief Returns winSigma value
                    */
                    CV_WRAP double getWinSigma() const;

                    /**@example samples/cpp/peopledetect.cpp
                    */
                    /**@brief Sets coefficients for the linear SVM classifier.
                    @param svmdetector coefficients for the linear SVM classifier.
                    */
                    CV_WRAP virtual void setSVMDetector(InputArray svmdetector);

                    /** @brief Reads HOGDescriptor parameters from a cv::FileNode.
                    @param fn File node
                    */
                    virtual bool read(FileNode& fn);

                    /** @brief Stores HOGDescriptor parameters in a cv::FileStorage.
                    @param fs File storage
                    @param objname Object name
                    */
                    virtual void write(FileStorage& fs, const String& objname) const;

                    /** @brief loads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file.
                    @param filename Path of the file to read.
                    @param objname The optional name of the node to read (if empty, the first top-level node will be used).
                    */
                    CV_WRAP virtual bool load(const String& filename, const String& objname = String());

                    /** @brief saves HOGDescriptor parameters and coefficients for the linear SVM classifier to a file
                    @param filename File name
                    @param objname Object name
                    */
                    CV_WRAP virtual void save(const String& filename, const String& objname = String()) const;

                    /** @brief clones the HOGDescriptor
                    @param c cloned HOGDescriptor
                    */
                    virtual void copyTo(HOGDescriptor& c) const;

                    /**@example samples/cpp/train_HOG.cpp
                    */
                    /** @brief Computes HOG descriptors of given image.
                    @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
                    @param descriptors Matrix of the type CV_32F
                    @param winStride Window stride. It must be a multiple of block stride.
                    @param padding Padding
                    @param locations Vector of Point
                    */
                    CV_WRAP virtual void compute(InputArray img,
                    CV_OUT std::vector<float>& descriptors,
                    Size winStride = Size(), Size padding = Size(),
                    const std::vector<Point>& locations = std::vector<Point>()) const;

                    /** @brief Performs object detection without a multi-scale window.
                    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
                    @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
                    @param weights Vector that will contain confidence values for each detected object.
                    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
                    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
                    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
                    @param winStride Window stride. It must be a multiple of block stride.
                    @param padding Padding
                    @param searchLocations Vector of Point includes set of requested locations to be evaluated.
                    */
                    CV_WRAP virtual void detect(InputArray img, CV_OUT std::vector<Point>& foundLocations,
                    CV_OUT std::vector<double>& weights,
                    double hitThreshold = 0, Size winStride = Size(),
                    Size padding = Size(),
                    const std::vector<Point>& searchLocations = std::vector<Point>()) const;

                    /** @brief Performs object detection without a multi-scale window.
                    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
                    @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
                    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
                    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
                    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
                    @param winStride Window stride. It must be a multiple of block stride.
                    @param padding Padding
                    @param searchLocations Vector of Point includes locations to search.
                    */
                    virtual void detect(InputArray img, CV_OUT std::vector<Point>& foundLocations,
                    double hitThreshold = 0, Size winStride = Size(),
                    Size padding = Size(),
                    const std::vector<Point>& searchLocations=std::vector<Point>()) const;

                    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
                    of rectangles.
                    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
                    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
                    @param foundWeights Vector that will contain confidence values for each detected object.
                    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
                    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
                    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
                    @param winStride Window stride. It must be a multiple of block stride.
                    @param padding Padding
                    @param scale Coefficient of the detection window increase.
                    @param finalThreshold Final threshold
                    @param useMeanshiftGrouping indicates grouping algorithm
                    */
                    CV_WRAP virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                    CV_OUT std::vector<double>& foundWeights, double hitThreshold = 0,
                    Size winStride = Size(), Size padding = Size(), double scale = 1.05,
                    double finalThreshold = 2.0,bool useMeanshiftGrouping = false) const;

                    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
                    of rectangles.
                    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
                    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
                    @param hitThreshold Threshold for the distance between features and SVM classifying plane.
                    Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient).
                    But if the free coefficient is omitted (which is allowed), you can specify it manually here.
                    @param winStride Window stride. It must be a multiple of block stride.
                    @param padding Padding
                    @param scale Coefficient of the detection window increase.
                    @param finalThreshold Final threshold
                    @param useMeanshiftGrouping indicates grouping algorithm
                    */
                    virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                    double hitThreshold = 0, Size winStride = Size(),
                    Size padding = Size(), double scale = 1.05,
                    double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;

                    /** @brief  Computes gradients and quantized gradient orientations.
                    @param img Matrix contains the image to be computed
                    @param grad Matrix of type CV_32FC2 contains computed gradients
                    @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
                    @param paddingTL Padding from top-left
                    @param paddingBR Padding from bottom-right
                    */
                    CV_WRAP virtual void computeGradient(InputArray img, InputOutputArray grad, InputOutputArray angleOfs,
                    Size paddingTL = Size(), Size paddingBR = Size()) const;

                    /** @brief Returns coefficients of the classifier trained for people detection (for 64x128 windows).
                    */
                    CV_WRAP static std::vector<float> getDefaultPeopleDetector();

                    /**@example samples/tapi/hog.cpp
                    */
                    /** @brief Returns coefficients of the classifier trained for people detection (for 48x96 windows).
                    */
                    CV_WRAP static std::vector<float> getDaimlerPeopleDetector();

                    //! Detection window size. Align to block size and block stride. Default value is Size(64,128).
                    CV_PROP Size winSize;

                    //! Block size in pixels. Align to cell size. Default value is Size(16,16).
                    CV_PROP Size blockSize;

                    //! Block stride. It must be a multiple of cell size. Default value is Size(8,8).
                    CV_PROP Size blockStride;

                    //! Cell size. Default value is Size(8,8).
                    CV_PROP Size cellSize;

                    //! Number of bins used in the calculation of histogram of gradients. Default value is 9.
                    CV_PROP int nbins;

                    //! not documented
                    CV_PROP int derivAperture;

                    //! Gaussian smoothing window parameter.
                    CV_PROP double winSigma;

                    //! histogramNormType
                    CV_PROP HOGDescriptor::HistogramNormType histogramNormType;

                    //! L2-Hys normalization method shrinkage.
                    CV_PROP double L2HysThreshold;

                    //! Flag to specify whether the gamma correction preprocessing is required or not.
                    CV_PROP bool gammaCorrection;

                    //! coefficients for the linear SVM classifier.
                    CV_PROP std::vector<float> svmDetector;

                    //! coefficients for the linear SVM classifier used when OpenCL is enabled
                    UMat oclSvmDetector;

                    //! not documented
                    float free_coef;

                    //! Maximum number of detection window increases. Default value is 64
                    CV_PROP int nlevels;

                    //! Indicates signed gradient will be used or not
                    CV_PROP bool signedGradient;

                    /** @brief evaluate specified ROI and return confidence value for each location
                    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
                    @param locations Vector of Point
                    @param foundLocations Vector of Point where each Point is detected object's top-left point.
                    @param confidences confidences
                    @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually
                    it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if
                    the free coefficient is omitted (which is allowed), you can specify it manually here
                    @param winStride winStride
                    @param padding padding
                    */
                    virtual void detectROI(InputArray img, const std::vector<cv::Point> &locations,
                    CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
                    double hitThreshold = 0, cv::Size winStride = Size(),
                    cv::Size padding = Size()) const;

                    /** @brief evaluate specified ROI and return confidence value for each location in multiple scales
                    @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
                    @param foundLocations Vector of rectangles where each rectangle contains the detected object.
                    @param locations Vector of DetectionROI
                    @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specified
                    in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.
                    @param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
                    */
                    virtual void detectMultiScaleROI(InputArray img,
                    CV_OUT std::vector<cv::Rect>& foundLocations,
                    std::vector<DetectionROI>& locations,
                    double hitThreshold = 0,
                    int groupThreshold = 0) const;

                    /** @brief Groups the object candidate rectangles.
                    @param rectList  Input/output vector of rectangles. Output vector includes retained and grouped rectangles. (The Python list is not modified in place.)
                    @param weights Input/output vector of weights of rectangles. Output vector includes weights of retained and grouped rectangles. (The Python list is not modified in place.)
                    @param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
                    @param eps Relative difference between sides of the rectangles to merge them into a group.
                    */
                    void groupRectangles(std::vector<cv::Rect>& rectList, std::vector<double>& weights, int groupThreshold, double eps) const;
            };


}


#endif //KALMAN_HOGDESC_H
