#include "main.h"

#include <stdio.h>
#include <string.h>
#include <ctime>
#include <math.h>
#include <stdexcept>
    
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <direct.h>
#endif

#include <string>
#include <iostream>

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check

#include "globalstate.h"
#include "gipuma.h"

#include "cameraGeometryUtils.h"
#include "mathUtils.h"
#include "GipumaMain.h"
#include "algorithmparameters.h"

/* process command line arguments
 * Input: argc, argv - command line arguments
 * Output: inputFiles, outputFiles, parameters, gt_parameters, - algorithm parameters
 */
static void getParametersFromFile(const char* filename, AlgorithmParameters& algParams)
{
    const std::string box_hsize = "box_hsize:";
    const std::string box_vsize = "box_vsize:";
    const std::string cost_gamma_opt = "cost_gamma:";
    const std::string num_iterations_opt = "iterations:";
    const std::string n_best_opt = "n_best:";
    const std::string cost_comb_opt = "cost_comb:";
    const std::string depth_min_opt = "depth_min:";
    const std::string depth_max_opt = "depth_max:";
    const std::string min_angle_opt = "min_angle:";
    const std::string max_angle_opt = "max_angle:";

    std::ifstream infile(filename, std::ifstream::in);
    if (!infile) {
        std::cout << std::endl << "fail to open parameters config file and will use default parameters.";
        return;
    }

    std::string line;

    while(std::getline(infile, line))
    {
        std::stringstream ss(line);
        std::string tmp;
        ss >> tmp;

        if (tmp == box_hsize) {
            int opt;
            ss >> opt;
            algParams.box_hsize = opt;
        } else if (tmp == box_vsize) {
            int opt;
            ss >> opt;
            algParams.box_vsize = opt;
        } else if (tmp == cost_gamma_opt) {
            float opt;
            ss >> opt;
            algParams.gamma = opt;
        } else if (tmp == num_iterations_opt) {
            int opt;
            ss >> opt;
            algParams.iterations = opt;
        } else if (tmp == n_best_opt) {
            int opt;
            ss >> opt;
            algParams.n_best = opt;
        } else if (tmp == cost_comb_opt) {
            int opt;
            ss >> opt;
            algParams.cost_comb = opt;
        } else if (tmp == depth_min_opt) {
            float opt;
            ss >> opt;
            algParams.depthMin = opt;
        } else if (tmp == depth_max_opt) {
            float opt;
            ss >> opt;
            algParams.depthMax = opt;
        } else if (tmp == min_angle_opt) {
            float opt;
            ss >> opt;
            algParams.min_angle = opt;
        } else if (tmp == max_angle_opt) {
            float opt;
            ss >> opt;
            algParams.max_angle = opt;
        } else {
            continue;
        }
    }

    return;
}

static void selectViews (CameraParameters &cameraParams, int imgWidth, int imgHeight, AlgorithmParameters &algParams ) {
    std::vector<Camera> &cameras = cameraParams.cameras;
    Camera ref = cameras[cameraParams.idRef];

    int x = imgWidth / 2;
    int y = imgHeight / 2;

    cameraParams.viewSelectionSubset.clear ();

    cv::Vec3f viewVectorRef = getViewVector ( ref, x, y);

    // TODO hardcoded value makes it a parameter
    float minimum_angle_degree = algParams.min_angle;
    float maximum_angle_degree = algParams.max_angle;

    unsigned int maximum_view = algParams.max_views;
    float minimum_angle_radians = minimum_angle_degree * M_PI / 180.0f;
    float maximum_angle_radians = maximum_angle_degree * M_PI / 180.0f;
    float min_depth = 9999;
    float max_depth = 0;
    if ( algParams.viewSelection )
        printf("Accepting intersection angle of central rays from %f to %f degrees, use --min_angle=<angle> and --max_angle=<angle> to modify them\n", minimum_angle_degree, maximum_angle_degree);
    for ( size_t i = 1; i < cameras.size (); i++ ) {
        //if ( !algParams.viewSelection ) { //select all views, dont perform selection
            //cameraParams.viewSelectionSubset.push_back ( i );
            //continue;
        //}

        cv::Vec3f vec = getViewVector ( cameras[i], x, y);

        float baseline = norm (cameras[0].C, cameras[i].C);
        float angle = getAngle ( viewVectorRef, vec );
        if ( angle > minimum_angle_radians &&
             angle < maximum_angle_radians ) //0.6 select if angle between 5.7 and 34.8 (0.6) degrees (10 and 30 degrees suggested by some paper)
        {
            if ( algParams.viewSelection ) {
                cameraParams.viewSelectionSubset.push_back ( i );
                //printf("\taccepting camera %ld with angle\t %f degree (%f radians) and baseline %f\n", i, angle*180.0f/M_PI, angle, baseline);
            }
            float min_range = (baseline/2.0f) / sin(maximum_angle_radians/2.0f);
            float max_range = (baseline/2.0f) / sin(minimum_angle_radians/2.0f);
            min_depth = std::min(min_range, min_depth);
            max_depth = std::max(max_range, max_depth);
            //printf("Min max ranges are %f %f\n", min_range, max_range);
            //printf("Min max depth are %f %f\n", min_depth, max_depth);
        }
        //else
            //printf("Discarding camera %ld with angle\t %f degree (%f radians) and baseline, %f\n", i, angle*180.0f/M_PI, angle, baseline);
    }

    if (algParams.depthMin == -1)
        algParams.depthMin = min_depth;
    if (algParams.depthMax == -1)
        algParams.depthMax = max_depth;

    if (!algParams.viewSelection) {
        cameraParams.viewSelectionSubset.clear();
        for ( size_t i = 1; i < cameras.size (); i++ )
            cameraParams.viewSelectionSubset.push_back ( i );
        return;
    }
    if (cameraParams.viewSelectionSubset.size() >= maximum_view) {
        printf("Too many camera, randomly selecting only %d of them (modify with --max_views=<number>)\n", maximum_view);
        std::srand ( unsigned ( time(0) ) );
        std::random_shuffle( cameraParams.viewSelectionSubset.begin(), cameraParams.viewSelectionSubset.end() ); // shuffle elements of v
        cameraParams.viewSelectionSubset.erase (cameraParams.viewSelectionSubset.begin()+maximum_view,cameraParams.viewSelectionSubset.end());
    }
    //for (auto i : cameraParams.viewSelectionSubset )
        //printf("\taccepting camera %d\n", i);
}

static void delTexture (int num, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (int i=0; i<num; i++) {
        cudaFreeArray(cuArray[i]);
        cudaDestroyTextureObject(texs[i]);
    }
}

static void addImageToTextureUint (std::vector<cv::Mat_<uint8_t> > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with uint8_t point type
        cudaChannelFormatDesc channelDesc =
        //cudaCreateChannelDesc (8,
                               //0,
                               //0,
                               //0,
                               //cudaChannelFormatKindUnsigned);
        cudaCreateChannelDesc<char>();
        // Allocate array with correct size and number of channels
        checkCudaErrors(cudaMallocArray(&cuArray[i],
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray[i],
                                              0,
                                              0,
                                              imgs[i].ptr<uint8_t>(),
                                              imgs[i].step[0],
                                              cols*sizeof(uint8_t),
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModePoint;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        //texs[i] = texObj;
    }
    return;
}
static void addImageToTextureFloatColor (std::vector<cv::Mat > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Allocate array with correct size and number of channels
        //cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray[i],
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray[i],
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float)*4,
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
    }
    return;
}

static void addImageToTextureFloatGray (std::vector<cv::Mat > &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc (32,
                               0,
                               0,
                               0,
                               cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        checkCudaErrors(cudaMallocArray(&cuArray[i],
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray[i],
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float),
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        //texs[i] = texObj;
    }
    return;
}

static void selectCudaDevice ()
{
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "There is no cuda capable device!\n");
        exit(EXIT_FAILURE);
    } 
    std::cout << std::endl << "Detected " << deviceCount << " devices!" << std::endl;
    std::vector<int> usableDevices;
    std::vector<std::string> usableDeviceNames;
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 3 && prop.minor >= 0) {
                usableDevices.push_back(i);
                usableDeviceNames.push_back(std::string(prop.name));
            } else {
                std::cout << "CUDA capable device " << std::string(prop.name)
                     << " is only compute cabability " << prop.major << '.'
                     << prop.minor << std::endl;
            }
        } else {
            std::cout << "Could not check device properties for one of the cuda "
                    "devices!" << std::endl;
        }
    }
    if(usableDevices.empty()) {
        fprintf(stderr, "There is no cuda device supporting gipuma!\n");
        exit(EXIT_FAILURE);
    }
    std::cout << "Detected gipuma compatible device: " << usableDeviceNames[0] << std::endl;;
    checkCudaErrors(cudaSetDevice(usableDevices[0]));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*128);
}

bool runGipuma(const std::vector<cv::Mat_<float>> &images,
                     const std::vector<cv::Mat_<float>> &projection_matrices,
                     cv::Mat_<float> &depth_map,
                     cv::Mat_<cv::Point3_<float>> &normal_map,
                     float &dMin, float &dMax,
                     AlgorithmParameters& algParams)
{
    size_t numImages = images.size();
    algParams.num_img_processed = std::min((int)numImages, algParams.num_img_processed);

    std::vector<cv::Mat_<float> > img_grayscale_float(numImages);

    for ( size_t i = 0; i < numImages; i++ ) {
        img_grayscale_float[i] = images[i];
        if ( algParams.color_processing ) {
            std::cout << "process colorMap doesn't implement temporarily. " << std::endl;
        }

        if ( img_grayscale_float[i].rows == 0 ) {
            printf ( "Image seems to be invalid\n" );
            return false;
        }
    }

    uint32_t rows = img_grayscale_float[0].rows;
    uint32_t cols = img_grayscale_float[0].cols;

    GlobalState *gs = new GlobalState;

    CameraParameters cameraParams = getCameraParameters(*(gs->cameras), projection_matrices, algParams.cam_scale);

    //allocation for disparity and normal stores
    std::vector<cv::Mat_<float> > disp(algParams.num_img_processed);
    for ( int i = 0; i < algParams.num_img_processed; i++ ) {
        disp[i] = cv::Mat::zeros ( img_grayscale_float[0].rows, img_grayscale_float[0].cols, CV_32F );
    }

    selectViews ( cameraParams, cols, rows, algParams);
    int numSelViews = cameraParams.viewSelectionSubset.size ();
    for ( int i = 0; i < numSelViews; i++ ) {
        gs->cameras->viewSelectionSubset[i] = cameraParams.viewSelectionSubset[i];
    }

    for ( int i = 0; i < algParams.num_img_processed; i++ ) {
        cameraParams.cameras[i].depthMin = algParams.depthMin;
        cameraParams.cameras[i].depthMax = algParams.depthMax;

        gs->cameras->cameras[i].depthMin = algParams.depthMin;
        gs->cameras->cameras[i].depthMax = algParams.depthMax;

        algParams.min_disparity = disparityDepthConversion ( cameraParams.f, cameraParams.cameras[i].baseline, cameraParams.cameras[i].depthMax );
        algParams.max_disparity = disparityDepthConversion ( cameraParams.f, cameraParams.cameras[i].baseline, cameraParams.cameras[i].depthMin );


        double minVal, maxVal;
        minMaxLoc(disp[i], &minVal, &maxVal);
    }
    dMin = algParams.depthMin;
    dMax = algParams.depthMax;

    // run gpu run
    // Init parameters
    gs->params = &algParams;

    gs->cameras->viewSelectionSubsetNumber = numSelViews;

    // Init ImageInfo
    gs->cameras->cols = cols;
    gs->cameras->rows = rows;
    gs->params->cols = cols;
    gs->params->rows = rows;

    // Resize lines
    {
        gs->lines->n = rows * cols;
        gs->lines->resize(rows * cols);
        //gs->lines.s = img_grayscale[0].step[0];
        gs->lines->s = cols;
        gs->lines->l = cols;
    }

    std::vector<cv::Mat > img_grayscale_float_new(numImages);
    for (size_t i = 0; i<numImages; i++) {
        img_grayscale_float[i].convertTo(img_grayscale_float_new[i], CV_32FC1);
    }

    addImageToTextureFloatGray(img_grayscale_float_new, gs->imgs, gs->cuArray);

    runcuda(*gs);
    for(size_t i = 0; i < cols; i++ )
        for(size_t j = 0; j < rows; j++ )
        {
            int center = i+cols*j;
            float4 n = gs->lines->norm4[center];
            normal_map(j, i) = cv::Point3_<float>(n.x,
                                             n.y,
                                             n.z);
            depth_map(j, i) = gs->lines->norm4[center].w;
        }

    // Free memory
    delTexture(algParams.num_img_processed, gs->imgs, gs->cuArray);
    delete gs;
    delete &algParams;
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return true;
}

bool GipumaMain(const std::vector<cv::Mat_<float>> &images,
               const std::vector<cv::Mat_<float>> &projection_matrices,
               cv::Mat_<float> &depth_map,
               cv::Mat_<cv::Point3_<float>> &normal_map,
               float &dMin, float &dMax,
               const char* config_filename)
{
    AlgorithmParameters* algParams = new AlgorithmParameters;

    getParametersFromFile(config_filename, *algParams);

    bool ret = runGipuma(images, projection_matrices, depth_map, normal_map, dMin, dMax, *algParams);

    return ret;
}