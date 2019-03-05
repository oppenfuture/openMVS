include_directories(.)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_75,code=sm_75)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_50,code=sm_50)


if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -pedantic -Wno-unused-function -Wno-switch")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -Ofast -ffast-math -march=native") # extend release-profile with fast-math
endif()


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

SET(LIBGIPUMA_SRC
    cameraGeometryUtils.cpp
    vector_operations.h
    camera.h
    globalstate.h
    helper_string.h
    helper_cuda.h
    algorithmparameters.h
    cameraparameters.h
    linestate.h
    main.h
    mathUtils.cpp
    gipuma.cu
    GipumaMain.cpp)

ADD_LIBRARY(gipuma STATIC ${LIBGIPUMA_SRC})
