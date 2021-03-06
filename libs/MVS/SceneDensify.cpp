/*
* SceneDensify.cpp
*
* Copyright (c) 2014-2015 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#include "Common.h"
#include "Scene.h"
// MRF: view selection
#include "../Math/TRWS/MRFEnergy.h"
// CGAL: depth-map initialization
#include <cuda_runtime.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <nlohmann/json.hpp>
#include "../Gipuma/GipumaMain.h"

using namespace MVS;

// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DENSE_USE_OPENMP
#endif


// S T R U C T S ///////////////////////////////////////////////////

// Dense3D data.events
enum EVENT_TYPE {
	EVT_FAIL = 0,
	EVT_CLOSE,

	EVT_PROCESSIMAGE,

	EVT_ESTIMATEDEPTHMAP,
	EVT_OPTIMIZEDEPTHMAP,
	EVT_SAVEDEPTHMAP,

	EVT_FILTERDEPTHMAP,
	EVT_ADJUSTDEPTHMAP,
};

class EVTFail : public Event
{
public:
	EVTFail() : Event(EVT_FAIL) {}
};
class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};

class EVTProcessImage : public Event
{
public:
	IIndex idxImage;
	EVTProcessImage(IIndex _idxImage) : Event(EVT_PROCESSIMAGE), idxImage(_idxImage) {}
};

class EVTEstimateDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTEstimateDepthMap(IIndex _idxImage) : Event(EVT_ESTIMATEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTOptimizeDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTOptimizeDepthMap(IIndex _idxImage) : Event(EVT_OPTIMIZEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTSaveDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTSaveDepthMap(IIndex _idxImage) : Event(EVT_SAVEDEPTHMAP), idxImage(_idxImage) {}
};

class EVTFilterDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTFilterDepthMap(IIndex _idxImage) : Event(EVT_FILTERDEPTHMAP), idxImage(_idxImage) {}
};
class EVTAdjustDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTAdjustDepthMap(IIndex _idxImage) : Event(EVT_ADJUSTDEPTHMAP), idxImage(_idxImage) {}
};
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

// structure used to compute all depth-maps
class DepthMapsData
{
public:
	DepthMapsData(Scene& _scene);
	~DepthMapsData();

	bool SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap);
	bool SelectViews(DepthData& depthData);
	bool InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors);
	bool InitDepthMap(DepthData& depthData);
	bool EstimateDepthMap(IIndex idxImage);
	bool CpuPatchMatch(IIndex idxImage, cList<SEACAVE::Thread> &threads, Image64F imageSum0, cList<DepthEstimator> &estimators);
	bool GipumaPatchMatch(IIndex idxImage, const Image8U::Size &size, DepthData &depthData);

	bool RemoveSmallSegments(DepthData& depthData);
	bool GapInterpolation(DepthData& depthData);

	bool FilterDepthMap(DepthData& depthData, const IIndexArr& idxNeighbors, bool bAdjust=true);
	void FuseDepthMaps(PointCloud& pointcloud, bool bEstimateNormal);

	bool InitDepthMapWithoutTriangulation(DepthData& depthData, const Image8U::Size size, const bool updateFlag=false);
	bool InitDepthMapByRealsense(DepthData& depthData, const std::string& realsense_filename, const Image &image);

protected:

	static void* STCALL ScoreDepthMapTmp(void*);
	static void* STCALL EstimateDepthMapTmp(void*);
	static void* STCALL EndDepthMapTmp(void*);

public:
	Scene& scene;

	DepthDataArr arrDepthData;

	// used internally to estimate the depth-maps
	// Image8U::Size prevDepthMapSize; // remember the size of the last estimated depth-map
	// Image8U::Size prevDepthMapSizeTrg; // ... same for target image
	// DepthEstimator::MapRefArr coords; // map pixel index to zigzag matrix coordinates
	// DepthEstimator::MapRefArr coordsTrg; // ... same for target image
};
/*----------------------------------------------------------------*/

/******************************************************************/
namespace MVS {
	cv::Mat_<float> MatxtoMat(const cv::Matx<double, 3, 4> &matx34d) {
		cv::Mat_<float> out = cv::Mat::zeros(3, 4, CV_32FC1);
		for (int c = 0; c < 4; ++c) {
			for (int r = 0; r < 3; ++r) {
				out(r, c) = matx34d(r, c);
			}
		}

		return out;
	}

	TImage<TPoint3<float> > TransformTImage(const cv::Mat_<cv::Point3_<float> > &M) {
		TImage<TPoint3<float> > out(M.rows, M.cols);
		for (int c = 0; c < M.cols; ++c) {
			for (int r = 0; r < M.rows; ++r) {
				cv::Point3_<float> m = M(r, c);
				TPoint3<float> n(m.x, m.y, m.z);
				out(r, c) = n;
			}
		}
		return out;
	}

	inline std::string GetDir(const std::string& path) {
		std::size_t botDirPos = path.find_last_of("/");
		std::string dir = path.substr(0, botDirPos+1);

		return dir;
	}

	inline std::string GetFile(const std::string& path) {
		std::size_t botDirPos = path.find_last_of("/");
		std::string file = path.substr(botDirPos+1, path.size());

		return file;
	}

	inline TPoint3<float> TransformPointI2C(const TPoint3<float>& X, const cv::Matx<float,3,3>& K){
		return TPoint3<float>(
				float((X.x-K(0,2))*X.z/K(0,0)),
				float((X.y-K(1,2))*X.z/K(1,1)),
				float(X.z));
	}

	inline TPoint3<float> TransformPointC2W(const TPoint3<float>& X, const cv::Matx<float,3,3>& R, const cv::Point3_<float>& C){
		return R.t() * X + C;
	}

	inline TPoint3<float> TransformPointI2W(const TPoint3<float>& X, const cv::Matx<float,3,3>& K, const cv::Matx<float,3,3>& R, const cv::Point3_<float>& C){
		return TransformPointC2W(TransformPointI2C(X, K), R, C);
	}

  void UpdateZ(int x, int y, int width, int height, const TPoint3<float>& pt,
    std::vector<float> &weights, std::vector<float> &z_vals, std::vector<float> &z_buffer) {
    if (x < 0 || x >= width || y < 0 || y >= height)
      return;
    float epsilon = 0.05f; // 5cm
    int idx = x + y * width;
    if (pt.z > z_buffer[idx] + epsilon) {
      // point farther from view then z_buffer should be ignore
      return;
    } else if (pt.z + epsilon < z_buffer[idx]) {
      // update z_buffer with closer point, and clear z_vals and weights;
      z_buffer[idx] = pt.z;
      z_vals[idx] = 0.f;
      weights[idx] = 0.f;
    }
    float w = std::fabs(pt.x - x) * std::fabs(pt.y - y);
    weights[idx] += w;
    z_vals[idx] += w * pt.z;
  }

  void WarpDepth(const cv::Matx<float,3,3>& K, const cv::Matx<float,3,3>& R, const cv::Point3_<float>& C, const Image& image, DepthData &depth_data) {
    std::vector<TPoint3<float> > image_points;
    std::vector<TPoint3<float> > world_points;
    std::vector<TPoint3<float> > rgb_points;
    const Camera& camera = image.camera;

    for (auto r=0; r<depth_data.depthMap.height(); ++r) {
      for (auto c=0; c<depth_data.depthMap.width(); ++c) {
        if (depth_data.depthMap(r, c) != 0) {
          image_points.push_back(TPoint3<float>(c, r, depth_data.depthMap(r, c)));
        }
      }
    }

    for (auto &itr: image_points) {
      world_points.push_back(MVS::TransformPointI2W(itr, K, R, C));
    }

    for (auto &itr: world_points) {
      TPoint3<double> it = itr;
      TPoint3<double> X = camera.TransformPointW2C(it);
      TPoint2<float> tmp = camera.TransformPointC2I(X);
      TPoint3<float> x = TPoint3<float>(tmp.x, tmp.y, X.z);
      rgb_points.push_back(x);
    }

    std::vector<float> weights(image.width*image.height, 0.f);
    std::vector<float> z_vals(image.width * image.height, 0.f);
    std::vector<float> z_buffer(image.width * image.height, 1000.f);


    for (auto &pt: rgb_points) {
      int x0 = int(pt.x);
      int x1 = x0 + 1;
      int y0 = int(pt.y);
      int y1 = y0 + 1;
      UpdateZ(x0, y0, image.width, image.height, pt, weights, z_vals, z_buffer);
      UpdateZ(x1, y0, image.width, image.height, pt, weights, z_vals, z_buffer);
      UpdateZ(x0, y1, image.width, image.height, pt, weights, z_vals, z_buffer);
      UpdateZ(x1, y1, image.width, image.height, pt, weights, z_vals, z_buffer);
    }

    depth_data.depthMap = cv::Mat_<float>::zeros(image.height, image.width);
		depth_data.dMin = FLT_MAX;
		depth_data.dMax = 0;

    for (size_t r = 0; r < image.height; ++r) {
      for (size_t c = 0; c < image.width; ++c) {
        int idx = r * image.width + c;
        if (weights[idx] > 0) {
					float d = z_vals[idx] / weights[idx];
					if (depth_data.dMin > d)
						depth_data.dMin = d;
					if (depth_data.dMax < d)
						depth_data.dMax = d;
          depth_data.depthMap(r, c) = d;
        }
      }
    }
  #if TD_VERBOSE != TD_VERBOSE_OFF
  	if (g_nVerbosityLevel > 4) {
    	depth_data.depthMap.Save(image.name.substr(0, image.name.size()-4)+".pfm");
  	}
	#endif
  }
} // namespace MVS
/****************************************************************/

DepthMapsData::DepthMapsData(Scene& _scene)
	:
	scene(_scene),
	arrDepthData(_scene.images.GetSize())
{
} // constructor

DepthMapsData::~DepthMapsData()
{
} // destructor
/*----------------------------------------------------------------*/


// globally choose the best target view for each image,
// trying in the same time the selected image pairs to cover the whole scene;
// the map of selected neighbors for each image is returned in neighborsMap.
// For each view a list of neighbor views ordered by number of shared sparse points and overlapped image area is given.
// Next a graph is formed such that the vertices are the views and two vertices are connected by an edge if the two views have each other as neighbors.
// For each vertex, a list of possible labels is created using the list of neighbor views and scored accordingly (the score is normalized by the average score).
// For each existing edge, the score is defined such that pairing the same two views for any two vertices is discouraged (a constant high penalty is applied for such edges).
// This primal-dual defined problem, even if NP hard, can be solved by a Belief Propagation like algorithm, obtaining in general a solution close enough to optimality.
bool DepthMapsData::SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap)
{
	// find all pair of images valid for dense reconstruction
	typedef std::unordered_map<uint64_t,float> PairAreaMap;
	PairAreaMap edges;
	double totScore(0);
	unsigned numScores(0);
	FOREACH(i, images) {
		const IIndex idx(images[i]);
		ASSERT(imagesMap[idx] != NO_ID);
		const ViewScoreArr& neighbors(arrDepthData[idx].neighbors);
		ASSERT(neighbors.GetSize() <= OPTDENSE::nMaxViews);
		// register edges
		FOREACHPTR(pNeighbor, neighbors) {
			const IIndex idx2(pNeighbor->idx.ID);
			ASSERT(imagesMap[idx2] != NO_ID);
			edges[MakePairIdx(idx,idx2)] = pNeighbor->idx.area;
			totScore += pNeighbor->score;
			++numScores;
		}
	}
	if (edges.empty())
		return false;
	const float avgScore((float)(totScore/(double)numScores));

	// run global optimization
	const float fPairwiseMul = OPTDENSE::fPairwiseMul; // default 0.3
	const float fEmptyUnaryMult = 6.f;
	const float fEmptyPairwise = 8.f*OPTDENSE::fPairwiseMul;
	const float fSamePairwise = 24.f*OPTDENSE::fPairwiseMul;
	const IIndex _num_labels = OPTDENSE::nMaxViews+1; // N neighbors and an empty state
	const IIndex _num_nodes = images.GetSize();
	typedef MRFEnergy<TypeGeneral> MRFEnergyType;
	CAutoPtr<MRFEnergyType> energy(new MRFEnergyType(TypeGeneral::GlobalSize()));
	CAutoPtrArr<MRFEnergyType::NodeId> nodes(new MRFEnergyType::NodeId[_num_nodes]);
	typedef SEACAVE::cList<TypeGeneral::REAL, const TypeGeneral::REAL&, 0> EnergyCostArr;
	// unary costs: inverse proportional to the image pair score
	EnergyCostArr arrUnary(_num_labels);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		FOREACH(k, neighbors)
			arrUnary[k] = avgScore/neighbors[k].score; // use average score to normalize the values (not to depend so much on the number of features in the scene)
		arrUnary[neighbors.GetSize()] = fEmptyUnaryMult*(neighbors.IsEmpty()?avgScore*0.01f:arrUnary[neighbors.GetSize()-1]);
		nodes[n] = energy->AddNode(TypeGeneral::LocalSize(neighbors.GetSize()+1), TypeGeneral::NodeData(arrUnary.Begin()));
	}
	// pairwise costs: as ratios between the area to be covered and the area actually covered
	EnergyCostArr arrPairwise(_num_labels*_num_labels);
	for (PairAreaMap::const_reference edge: edges) {
		const PairIdx pair(edge.first);
		const float area(edge.second);
		const ViewScoreArr& neighborsI(arrDepthData[pair.i].neighbors);
		const ViewScoreArr& neighborsJ(arrDepthData[pair.j].neighbors);
		arrPairwise.Empty();
		FOREACHPTR(pNj, neighborsJ) {
			const IIndex i(pNj->idx.ID);
			const float areaJ(area/pNj->idx.area);
			FOREACHPTR(pNi, neighborsI) {
				const IIndex j(pNi->idx.ID);
				const float areaI(area/pNi->idx.area);
				arrPairwise.Insert(pair.i == i && pair.j == j ? fSamePairwise : fPairwiseMul*(areaI+areaJ));
			}
			arrPairwise.Insert(fEmptyPairwise+fPairwiseMul*areaJ);
		}
		for (const ViewScore& Ni: neighborsI) {
			const float areaI(area/Ni.idx.area);
			arrPairwise.Insert(fPairwiseMul*areaI+fEmptyPairwise);
		}
		arrPairwise.Insert(fEmptyPairwise*2);
		const IIndex nodeI(imagesMap[pair.i]);
		const IIndex nodeJ(imagesMap[pair.j]);
		energy->AddEdge(nodes[nodeI], nodes[nodeJ], TypeGeneral::EdgeData(TypeGeneral::GENERAL, arrPairwise.Begin()));
	}

	// minimize energy
	MRFEnergyType::Options options;
	options.m_eps = OPTDENSE::fOptimizerEps;
	options.m_iterMax = OPTDENSE::nOptimizerMaxIters;
	#ifndef _RELEASE
	options.m_printIter = 1;
	options.m_printMinIter = 1;
	#endif
	#if 1
	TypeGeneral::REAL energyVal, lowerBound;
	energy->Minimize_TRW_S(options, lowerBound, energyVal);
	#else
	TypeGeneral::REAL energyVal;
	energy->Minimize_BP(options, energyVal);
	#endif

	// extract optimized depth map
	neighborsMap.Resize(_num_nodes);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		IIndex& idxNeighbor = neighborsMap[n];
		const IIndex label((IIndex)energy->GetSolution(nodes[n]));
		ASSERT(label <= neighbors.GetSize());
		if (label == neighbors.GetSize()) {
			idxNeighbor = NO_ID; // empty
		} else {
			idxNeighbor = label;
			DEBUG_ULTIMATE("\treference image %3u paired with target image %3u (idx %2u)", images[n], neighbors[label].idx.ID, label);
		}
	}

	// remove all images with no valid neighbors
	RFOREACH(i, neighborsMap) {
		if (neighborsMap[i] == NO_ID) {
			// remove image with no neighbors
			for (IIndex& imageMap: imagesMap)
				if (imageMap != NO_ID && imageMap > i)
					--imageMap;
			imagesMap[images[i]] = NO_ID;
			images.RemoveAtMove(i);
			neighborsMap.RemoveAtMove(i);
		}
	}
	return !images.IsEmpty();
} // SelectViews
/*----------------------------------------------------------------*/

// compute visibility for the reference image (the first image in "images")
// and select the best views for reconstructing the depth-map;
// extract also all 3D points seen by the reference image
bool DepthMapsData::SelectViews(DepthData& depthData)
{
	// find and sort valid neighbor views
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(depthData.neighbors.IsEmpty());
	ASSERT(scene.images[idxImage].neighbors.IsEmpty());
	if (!scene.SelectNeighborViews(idxImage, depthData.points, OPTDENSE::nMinViews, OPTDENSE::nMinViewsTrustPoint>1?OPTDENSE::nMinViewsTrustPoint:2, FD2R(OPTDENSE::fOptimAngle)))
		return false;
	depthData.neighbors.CopyOf(scene.images[idxImage].neighbors);

	// remove invalid neighbor views
	const float fMinArea(OPTDENSE::fMinArea);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(FD2R(OPTDENSE::fMinAngle));
	const float fMaxAngle(FD2R(OPTDENSE::fMaxAngle));
	if (!Scene::FilterNeighborViews(depthData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, OPTDENSE::nMaxViews)) {
		DEBUG_EXTRA("error: reference image %3u has no good images in view", idxImage);
		return false;
	}
	return true;
} // SelectViews
/*----------------------------------------------------------------*/

// select target image for the reference image (the first image in "images")
// and initialize images data;
// if idxNeighbor is not NO_ID, only the reference image and the given neighbor are initialized;
// if numNeighbors is not 0, only the first numNeighbors neighbors are initialized;
// otherwise all are initialized;
// returns false if there are no good neighbors to estimate the depth-map
bool DepthMapsData::InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors)
{
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(!depthData.neighbors.IsEmpty());
	ASSERT(depthData.images.IsEmpty());

	// set this image the first image in the array
	depthData.images.Reserve(depthData.neighbors.GetSize()+1);
	depthData.images.AddEmpty();

	if (idxNeighbor != NO_ID) {
		// set target image as the given neighbor
		const ViewScore& neighbor = depthData.neighbors[idxNeighbor];
		DepthData::ViewData& imageTrg = depthData.images.AddEmpty();
		imageTrg.pImageData = &scene.images[neighbor.idx.ID];
		imageTrg.scale = neighbor.idx.scale;
		imageTrg.camera = imageTrg.pImageData->camera;
		imageTrg.pImageData->image.toGray(imageTrg.image, cv::COLOR_BGR2GRAY, true);
		if (imageTrg.ScaleImage(imageTrg.image, imageTrg.image, imageTrg.scale))
			imageTrg.camera = imageTrg.pImageData->GetCamera(scene.platforms, imageTrg.image.size());
		DEBUG_EXTRA("Reference image %3u paired with image %3u", idxImage, neighbor.idx.ID);
	} else {
		// init all neighbor views too (global reconstruction is used)
		const float fMinScore(MAXF(depthData.neighbors.First().score*(OPTDENSE::fViewMinScoreRatio*0.1f), OPTDENSE::fViewMinScore));
		FOREACH(idx, depthData.neighbors) {
			const ViewScore& neighbor = depthData.neighbors[idx];
			if ((numNeighbors && depthData.images.GetSize() > numNeighbors) ||
				(neighbor.score < fMinScore))
				break;
			DepthData::ViewData& imageTrg = depthData.images.AddEmpty();
			imageTrg.pImageData = &scene.images[neighbor.idx.ID];
			imageTrg.scale = neighbor.idx.scale;
			imageTrg.camera = imageTrg.pImageData->camera;
			imageTrg.pImageData->image.toGray(imageTrg.image, cv::COLOR_BGR2GRAY, true);
			if (imageTrg.ScaleImage(imageTrg.image, imageTrg.image, imageTrg.scale))
				imageTrg.camera = imageTrg.pImageData->GetCamera(scene.platforms, imageTrg.image.size());
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// print selected views
		if (g_nVerbosityLevel > 2) {
			String msg;
			for (IDX i=1; i<depthData.images.GetSize(); ++i)
				msg += String::FormatString(" %3u(%.2fscl)", depthData.images[i].pImageData-scene.images.Begin(), depthData.images[i].scale);
			VERBOSE("Reference image %3u paired with %u views:%s (%u shared points)", idxImage, depthData.images.GetSize()-1, msg.c_str(), depthData.points.GetSize());
		} else
		DEBUG_EXTRA("Reference image %3u paired with %u views", idxImage, depthData.images.GetSize()-1);
		#endif
	}
	if (depthData.images.GetSize() < 2) {
		depthData.images.Release();
		return false;
	}

	// init the first image as well
	DepthData::ViewData& imageRef = depthData.images.First();
	imageRef.scale = 1;
	imageRef.pImageData = &scene.images[idxImage];
	imageRef.pImageData->image.toGray(imageRef.image, cv::COLOR_BGR2GRAY, true);
	imageRef.camera = imageRef.pImageData->camera;
	// init mask
	const auto& maskImage = imageRef.pImageData->mask;
	if (!maskImage.empty()) {
		depthData.mask.create(maskImage.rows, maskImage.cols);
		for (int i = 0; i < maskImage.area(); ++i)
			depthData.mask.set(i, maskImage[i] > 0.5f);
	}
	DepthEstimator::MapMatrix2ZigzagIdx(imageRef.image.size(), depthData.coords, depthData.mask, MAXF(64,(int)scene.nMaxThreads*8));
	return true;
} // InitViews
/*----------------------------------------------------------------*/

namespace CGAL {
typedef CGAL::Simple_cartesian<double> kernel_t;
typedef CGAL::Projection_traits_xy_3<kernel_t> Geometry;
typedef CGAL::Delaunay_triangulation_2<Geometry> Delaunay;
typedef CGAL::Delaunay::Face_circulator FaceCirculator;
typedef CGAL::Delaunay::Face_handle FaceHandle;
typedef CGAL::Delaunay::Vertex_circulator VertexCirculator;
typedef CGAL::Delaunay::Vertex_handle VertexHandle;
typedef kernel_t::Point_3 Point;
}

// triangulate in-view points, generating a 2D mesh
// return also the estimated depth boundaries (min and max depth)
std::pair<float,float> TriangulatePointsDelaunay(CGAL::Delaunay& delaunay, const Scene& scene, const DepthData::ViewData& image, const IndexArr& points)
{
	ASSERT(sizeof(Point3) == sizeof(X3D));
	ASSERT(sizeof(Point3) == sizeof(CGAL::Point));
	std::pair<float,float> depthBounds(FLT_MAX, 0.f);
	FOREACH(p, points) {
		const PointCloud::Point& point = scene.pointcloud.points[points[p]];
		const Point3 ptCam(image.camera.TransformPointW2C(Cast<REAL>(point)));
		const Point2 ptImg(image.camera.TransformPointC2I(ptCam));
		delaunay.insert(CGAL::Point(ptImg.x, ptImg.y, ptCam.z));
		const Depth depth((float)ptCam.z);
		if (depthBounds.first > depth)
			depthBounds.first = depth;
		if (depthBounds.second < depth)
			depthBounds.second = depth;
	}
	// if full size depth-map requested
	if (OPTDENSE::bAddCorners) {
		typedef TIndexScore<float,float> DepthDist;
		typedef CLISTDEF0(DepthDist) DepthDistArr;
		typedef Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::InnerStride<2> > FloatMap;
		// add the four image corners at the average depth
		const CGAL::VertexHandle vcorners[] = {
			delaunay.insert(CGAL::Point(0, 0, image.pImageData->avgDepth)),
			delaunay.insert(CGAL::Point(image.image.width(), 0, image.pImageData->avgDepth)),
			delaunay.insert(CGAL::Point(0, image.image.height(), image.pImageData->avgDepth)),
			delaunay.insert(CGAL::Point(image.image.width(), image.image.height(), image.pImageData->avgDepth))
		};
		// compute average depth from the closest 3 directly connected faces,
		// weighted by the distance
		const size_t numPoints = 3;
		for (int i=0; i<4; ++i) {
			const CGAL::VertexHandle vcorner = vcorners[i];
			CGAL::FaceCirculator cfc(delaunay.incident_faces(vcorner));
			if (cfc == 0)
				continue; // normally this should never happen
			const CGAL::FaceCirculator done(cfc);
			Point3d& poszA = (Point3d&)vcorner->point();
			const Point2d& posA = reinterpret_cast<const Point2d&>(poszA);
			const Ray3d rayA(Point3d::ZERO, normalized(image.camera.TransformPointI2C(poszA)));
			DepthDistArr depths(0, numPoints);
			do {
				CGAL::FaceHandle fc(cfc->neighbor(cfc->index(vcorner)));
				if (fc == delaunay.infinite_face())
					continue;
				for (int j=0; j<4; ++j)
					if (fc->has_vertex(vcorners[j]))
						goto Continue;
				// compute the depth as the intersection of the corner ray with
				// the plane defined by the face's vertices
				{
				const Point3d& poszB0 = (const Point3d&)fc->vertex(0)->point();
				const Point3d& poszB1 = (const Point3d&)fc->vertex(1)->point();
				const Point3d& poszB2 = (const Point3d&)fc->vertex(2)->point();
				const Planed planeB(
					image.camera.TransformPointI2C(poszB0),
					image.camera.TransformPointI2C(poszB1),
					image.camera.TransformPointI2C(poszB2)
				);
				const Point3d poszB(rayA.Intersects(planeB));
				if (poszB.z <= 0)
					continue;
				const Point2d posB((reinterpret_cast<const Point2d&>(poszB0)+reinterpret_cast<const Point2d&>(poszB1)+reinterpret_cast<const Point2d&>(poszB2))/3.f);
				const REAL dist(norm(posB-posA));
				depths.StoreTop<numPoints>(DepthDist((float)poszB.z, 1.f/(float)dist));
				}
				Continue:;
			} while (++cfc != done);
			if (depths.GetSize() != numPoints)
				continue; // normally this should never happen
			FloatMap vecDists(&depths[0].score, numPoints);
			vecDists *= 1.f/vecDists.sum();
			FloatMap vecDepths(&depths[0].idx, numPoints);
			poszA.z = vecDepths.dot(vecDists);
		}
	}
	return depthBounds;
}

// roughly estimate depth and normal maps by triangulating the sparse point cloud
// and interpolating normal and depth for all pixels
bool DepthMapsData::InitDepthMap(DepthData& depthData)
{
	TD_TIMER_STARTD();

	ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
	const DepthData::ViewData& image(depthData.images.First());
	ASSERT(!image.image.empty());

	// triangulate in-view points
	CGAL::Delaunay delaunay;
	const std::pair<float,float> thDepth(TriangulatePointsDelaunay(delaunay, scene, image, depthData.points));
	depthData.dMin = thDepth.first*0.9f;
	depthData.dMax = thDepth.second*1.1f;

	// create rough depth-map by interpolating inside triangles
	const Camera& camera = image.camera;
	depthData.depthMap.create(image.image.size());
	depthData.normalMap.create(image.image.size());
	if (!OPTDENSE::bAddCorners) {
		depthData.depthMap.setTo(Depth(0));
		depthData.normalMap.setTo(0.f);
	}
	struct RasterDepthDataPlaneData {
		const Camera& P;
		const BitMatrix& mask;
		DepthMap& depthMap;
		NormalMap& normalMap;
		Point3f normal;
		Point3f normalPlane;
		inline void operator()(const ImageRef& pt) {
			if (!depthMap.isInside(pt) || !(mask.empty() || mask.isSet(pt)))
				return;
			const float z(INVERT(normalPlane.dot(P.TransformPointI2C(Point2f(pt)))));
			ASSERT(z > 0);
			depthMap(pt) = z;
			normalMap(pt) = normal;
		}
	};
	RasterDepthDataPlaneData data = {camera, depthData.mask, depthData.depthMap, depthData.normalMap};
	for (CGAL::Delaunay::Face_iterator it=delaunay.faces_begin(); it!=delaunay.faces_end(); ++it) {
		const CGAL::Delaunay::Face& face = *it;
		const Point3f i0((const Point3&)face.vertex(0)->point());
		const Point3f i1((const Point3&)face.vertex(1)->point());
		const Point3f i2((const Point3&)face.vertex(2)->point());
		// compute the plane defined by the 3 points
		const Point3f c0(camera.TransformPointI2C(i0));
		const Point3f c1(camera.TransformPointI2C(i1));
		const Point3f c2(camera.TransformPointI2C(i2));
		const Point3f edge1(c1-c0);
		const Point3f edge2(c2-c0);
		data.normal = normalized(edge2.cross(edge1));
		data.normalPlane = data.normal * INVERT(data.normal.dot(c0));
		// draw triangle and for each pixel compute depth as the ray intersection with the plane
		Image8U::RasterizeTriangle(reinterpret_cast<const Point2f&>(i2), reinterpret_cast<const Point2f&>(i1), reinterpret_cast<const Point2f&>(i0), data);
	}

	DEBUG_ULTIMATE("Depth-map %3u roughly estimated from %u sparse points: %dx%d (%s)", &depthData-arrDepthData.Begin(), depthData.points.GetSize(), image.image.width(), image.image.height(), TD_TIMER_GET_FMT().c_str());
	return true;
} // InitDepthMap
/*----------------------------------------------------------------*/

bool DepthMapsData::InitDepthMapByRealsense(DepthData& depthData, const std::string& realsense_filename, const Image& image) {
  std::ifstream infile(realsense_filename);
  if (!infile.good()) {
    std::cerr << "Failed opening realsense_warp_data json" << std::endl;
    return false;
  }
  /*
  realsense_warp_data.json format:
      {
          "P1001012.JPG": {
              "ref_depth": {
                  "filename": str,
                  "intrinsics": list,  // (3x3), row major
                  "extrinsics": list   // (4*4), list[:3, :3].T = R, list[:3, 3] = C
              }
          },
          ...
      }
  */
  auto image_name = GetFile(image.name);
  nlohmann::json j = nlohmann::json::parse(infile);
  auto view = j[image_name];
  auto view_params = view["ref_depth"];
  auto ref_filename = view_params["filename"];
	// load realsense depth
  depthData.depthMap.Load(GetDir(image.name) + ref_filename.get<std::string>());

  std::vector<std::vector<float> > intrinsics;
  std::vector<std::vector<float> > extrinsics;
  if (view_params.find("intrinsics") != view_params.end())
    intrinsics = view_params["intrinsics"].get<decltype(intrinsics)>();
  if (view_params.find("extrinsics") != view_params.end())
    extrinsics = view_params["extrinsics"].get<decltype(extrinsics)>();

  if (intrinsics.empty())
    intrinsics = {
      {1383.5141750796581, 0.0, 960.0},
      {0.0, 1383.5141750796581, 540.0},
      {0.0, 0.0, 1.0}
    };
  if (extrinsics.empty())
    extrinsics = {
      {1, 0, 0, 0},
      {0, 1, 0, 0},
      {0, 0, 1, 0},
      {0, 0, 0, 1},
    };
  cv::Matx<float,3,3> K;
  cv::Matx<float,3,3> R;
  cv::Point3_<float> C(extrinsics[0][3], extrinsics[1][3], extrinsics[2][3]);
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      K(i, j) = intrinsics[i][j];
      R(i, j) = extrinsics[j][i];
    }
  }
	// realsense-depth to rgb-depth
  WarpDepth(K, R, C, image, depthData);

  return true;
} // InitDepthMapByRealsense

// initialize the confidence map (NCC score map) with the score of the current estimates
void* STCALL DepthMapsData::ScoreDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		if (!estimator.PreparePixelPatch(x) || (!estimator.FillPixelPatch() && estimator.confMap0(x) < 0.5)) {
			estimator.depthMap0(x) = 0;
			estimator.normalMap0(x) = Normal::ZERO;
			estimator.confMap0(x) = DepthEstimator::EncodeScoreScale(2.f);
			continue;
		}
		Depth& depth = estimator.depthMap0(x);
		Normal& normal = estimator.normalMap0(x);
		const Normal viewDir(Cast<float>(static_cast<const Point3&>(estimator.X0)));
		if (depth <= 0) {
			// invalid
			depth = 0;
			normal = Normal::ZERO;
			estimator.confMap0(x) = DepthEstimator::EncodeScoreScale(2.f);
			continue;
		} else if (normal.dot(viewDir) >= 0) {
			// replace invalid normal with random values
			normal = DepthEstimator::RandomNormal(viewDir);
		}
		estimator.confMap0(x) = DepthEstimator::EncodeScoreScale(estimator.ScorePixel(depth, normal));
	}
	return NULL;
}
// run propagation and random refinement cycles
void* STCALL DepthMapsData::EstimateDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
		estimator.ProcessPixel(idx);
	return NULL;
}
// remove all estimates with too big score and invert confidence map
void* STCALL DepthMapsData::EndDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	const float fOptimAngle(FD2R(OPTDENSE::fOptimAngle));
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		Depth& depth = estimator.depthMap0(x);
		Normal& normal = estimator.normalMap0(x);
		float& conf = estimator.confMap0(x);
		const unsigned invScaleRange(DepthEstimator::DecodeScoreScale(conf));
		ASSERT(depth >= 0);
		// check if the score is good enough
		// and that the cross-estimates is close enough to the current estimate
		if (conf > OPTDENSE::fNCCThresholdKeep) {
			#if 1 // used if gap-interpolation is active
			conf = 0;
			normal = Normal::ZERO;
			#endif
			depth = 0;
		} else {
			#if 1
			FOREACH(i, estimator.images)
				estimator.scores[i] = ComputeAngle<REAL,float>(estimator.image0.camera.TransformPointI2W(Point3(x,depth)).ptr(), estimator.image0.camera.C.ptr(), estimator.images[i].view.camera.C.ptr());
			#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
			const float fCosAngle(estimator.scores.size() > 1 ? estimator.scores.GetNth(estimator.idxScore) : estimator.scores.front());
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MEAN
			const float fCosAngle(estimator.scores.mean());
			#else
			const float fCosAngle(estimator.scores.minCoeff());
			#endif
			const float wAngle(MINF(POW(ACOS(fCosAngle)/fOptimAngle,1.5f),1.f));
			#else
			const float wAngle(1.f);
			#endif
			#if 1
			conf = wAngle/MAXF(conf,1e-2f);
			#elif 1
			conf = wAngle/(depth*SQUARE(MAXF(conf,1e-2f)));
			#else
			conf = SQRT((float)invScaleRange)*wAngle/(depth*SQUARE(MAXF(conf,1e-2f)));
			#endif
		}
	}
	return NULL;
}

bool DepthMapsData::InitDepthMapWithoutTriangulation(DepthData& depthData, const Image8U::Size size, const bool updateFlag) {
	// compute depth range and initialize known depths
	const int nPixelArea = (int)OPTDENSE::nPixelArea; // half windows size around a pixel to be initialize with the known depth
	const Camera& camera = depthData.images.First().camera;
	if (!updateFlag)
	{
		depthData.dMin = FLT_MAX;
		depthData.dMax = 0;
	}
	else
	{
		depthData.dMin = depthData.dMin / 0.9f;
		depthData.dMax = depthData.dMax / 1.1f;
	}
	float small_depth = 0.01;
	FOREACHPTR(pPoint, depthData.points) {
		const PointCloud::Point& X = scene.pointcloud.points[*pPoint];
		const Point3 camX(camera.TransformPointW2C(Cast<REAL>(X)));
		const ImageRef x(ROUND2INT(camera.TransformPointC2I(camX)));
		const float d((float)camX.z);
		const ImageRef sx(MAXF(x.x-nPixelArea,0), MAXF(x.y-nPixelArea,0));
		const ImageRef ex(MINF(x.x+nPixelArea,size.width-1), MINF(x.y+nPixelArea,size.height-1));

		bool valid_coord = false;
		for (int y=sx.y; y<=ex.y; ++y)
			for (int x=sx.x; x<=ex.x; ++x)
				if (depthData.mask.empty() || depthData.mask.isSet(y,x))
				{
					if (updateFlag && depthData.depthMap(y,x) != 0.f && abs(depthData.depthMap(y,x) - d) > small_depth)
						continue;
					depthData.depthMap(y,x) = d;
					valid_coord = true;
				}
		if (valid_coord)
		{
			if (depthData.dMin > d)
				depthData.dMin = d;
			if (depthData.dMax < d)
				depthData.dMax = d;
		}
	}
	depthData.dMin *= 0.9f;
	depthData.dMax *= 1.1f;
	return true;
}

// cpu patch match:
// estimate depth-map using propagation and random refinement with NCC score
// as in: "Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes", S. Shen, 2013
// The implementations follows closely the paper, although there are some changes/additions.
// Given two views of the same scene, we note as the "reference image" the view for which a depth-map is reconstructed, and the "target image" the other view.
// As a first step, the whole depth-map is approximated by interpolating between the available sparse points.
// Next, the depth-map is passed from top/left to bottom/right corner and the opposite sens for each of the next steps.
// For each pixel, first the current depth estimate is replaced with its neighbor estimates if the NCC score is better.
// Second, the estimate is refined by trying random estimates around the current depth and normal values, keeping the one with the best score.
// The estimation can be stopped at any point, and usually 2-3 iterations are enough for convergence.
// For each pixel, the depth and normal are scored by computing the NCC score between the patch in the reference image and the wrapped patch in the target image, as dictated by the homography matrix defined by the current values to be estimate.
// In order to ensure some smoothness while locally estimating each pixel, a bonus is added to the NCC score if the estimate for this pixel is close to the estimates for the neighbor pixels.
// Optionally, the occluded pixels can be detected by extending the described iterations to the target image and removing the estimates that do not have similar values in both views.
//
// gipuma patch match:
// use gipuma to estimate depth-map
//
// no patch match:
// use initialization as the estimated depth-map
bool DepthMapsData::EstimateDepthMap(IIndex idxImage)
{
    TD_TIMER_STARTD();

    // initialize depth and normal maps
    DepthData& depthData(arrDepthData[idxImage]);
    ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
    const DepthData::ViewData& image(depthData.images.First());
    ASSERT(!image.image.empty() && !depthData.images[1].image.empty());
    const Image8U::Size size(image.image.size());
    depthData.normalMap.create(size);
    depthData.confMap.create(size);
    const unsigned nMaxThreads(scene.nMaxThreads);

    // initialize the depth-map
    if (OPTDENSE::importReferenceDepth) {
        // reference depth map already loaded
        if (OPTDENSE::updateReferenceDepthWithPoints) {
            InitDepthMapWithoutTriangulation(depthData, size, true);
        }
        if (depthData.depthMap.empty()) {
            std::cout << "Failed to import reference depth." << std::endl;
            exit(1);
        }
    } else {
    depthData.depthMap.create(size); depthData.depthMap.memset(0);
    if (OPTDENSE::algorithm == 1) {
        if (OPTDENSE::nMinViewsTrustPoint < 2) {
            InitDepthMapWithoutTriangulation(depthData, size);
        } else {
           // compute rough estimates using the sparse point-cloud
            InitDepthMap(depthData);
        }
    } else if(OPTDENSE::algorithm == 2) {
        std::cout << "NoPatchMatch must offer reference depth." << std::endl;
        exit(1);
    }
}

    if (OPTDENSE::algorithm == 0) {
        if (!GipumaPatchMatch(idxImage, size, depthData))
            return false;
    }
    else {
#if TD_VERBOSE != TD_VERBOSE_OFF
        // save rough depth map as image
        if (g_nVerbosityLevel > 4) {
            depthData.depthMap.Save(ComposeDepthFilePath(idxImage, "init.pfm"));
            ExportDepthMap(ComposeDepthFilePath(idxImage, "init.png"), depthData.depthMap);
            ExportNormalMap(ComposeDepthFilePath(idxImage, "init.normal.png"), depthData.normalMap);
            ExportPointCloud(ComposeDepthFilePath(idxImage, "init.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
        }
#endif
    }

	// init integral images and index to image-ref map for the reference data
	Image64F imageSum0;
	cv::integral(image.image, imageSum0, CV_64F);
	// if (prevDepthMapSize != size) {
	// 	prevDepthMapSize = size;
	// 	BitMatrix mask;
	// 	DepthEstimator::MapMatrix2ZigzagIdx(size, coords, mask, MAXF(64,(int)nMaxThreads*8));
	// }

	// init threads
	ASSERT(nMaxThreads > 0);
	cList<DepthEstimator> estimators;
	estimators.Reserve(nMaxThreads);
	cList<SEACAVE::Thread> threads;
	if (nMaxThreads > 1)
		threads.Resize(nMaxThreads-1); // current thread is also used
	volatile Thread::safe_t idxPixel;

	// compute the reference confidence map (NCC score map) with the score of the current estimates
	{
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(depthData, idxPixel, imageSum0, DepthEstimator::RB2LT);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(ScoreDepthMapTmp, &estimators[i]);
		ScoreDepthMapTmp(&estimators.Last());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();

		estimators.Release();
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// save rough depth map as image
		if (g_nVerbosityLevel > 4) {
			ExportDepthMap(ComposeDepthFilePath(idxImage, "score.depth.png"), depthData.depthMap);
			ExportNormalMap(ComposeDepthFilePath(idxImage, "score.normal.png"), depthData.normalMap);
			ExportPointCloud(ComposeDepthFilePath(idxImage, "score.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			ExportConfidenceMap(ComposeDepthFilePath(idxImage, "score.conf.png"), depthData.confMap);
			depthData.confMap.Save(ComposeDepthFilePath(idxImage, "score.conf.pfm"));
		}
		#endif
	}

	if (OPTDENSE::algorithm == 1)
		if (!CpuPatchMatch(idxImage, threads, imageSum0, estimators))
		    return false;

	if (g_nVerbosityLevel > 4) {
		const String path(ComposeDepthFilePath(image.pImageData-scene.images.Begin(), "iter"));
		depthData.depthMap.Save(ComposeDepthFilePath(idxImage, "iter.depth.pfm"));
		depthData.normalMap.Save(ComposeDepthFilePath(idxImage, "iter.normal.pfm"));
		depthData.confMap.Save(ComposeDepthFilePath(idxImage, "iter.conf.pfm"));
		ExportDepthMap(path+".png", depthData.depthMap);
		ExportNormalMap(path+".normal.png", depthData.normalMap);
		ExportPointCloud(path+".ply", *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
	}

	// remove all estimates with too big score and invert confidence map
	{
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(depthData, idxPixel, imageSum0, DepthEstimator::DIRS);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(EndDepthMapTmp, &estimators[i]);
		EndDepthMapTmp(&estimators.Last());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.Release();

	#if TD_VERBOSE != TD_VERBOSE_OFF
		// save reduce depth map as image
		if (g_nVerbosityLevel > 4) {
			ExportDepthMap(ComposeDepthFilePath(idxImage, "reduce.depth.png"), depthData.depthMap);
			ExportNormalMap(ComposeDepthFilePath(idxImage, "reduce.normal.png"), depthData.normalMap);
			ExportPointCloud(ComposeDepthFilePath(idxImage, "reduce.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			ExportConfidenceMap(ComposeDepthFilePath(idxImage, "reduce.conf.png"), depthData.confMap);
			depthData.confMap.Save(ComposeDepthFilePath(idxImage, "reduce.conf.pfm"));
		}
	#endif
	}

	DEBUG_EXTRA("Depth-map for image %3u %s: %dx%d (%s)", image.pImageData-scene.images.Begin(),
		depthData.images.GetSize() > 2 ?
			String::FormatString("estimated using %2u images", depthData.images.GetSize()-1).c_str() :
			String::FormatString("with image %3u estimated", depthData.images[1].pImageData-scene.images.Begin()).c_str(),
		size.width, size.height, TD_TIMER_GET_FMT().c_str());

	return true;
} // EstimateDepthMap
/*----------------------------------------------------------------*/

bool DepthMapsData::GipumaPatchMatch(IIndex idxImage, const Image8U::Size &size, DepthData &depthData)
{
    if (OPTDENSE::importReferenceDepth) {
        // use warped depth as init confMap
        cv::Mat mask_img(size.height, size.width, CV_32F);
        cv::threshold(depthData.depthMap, mask_img, 0.0, 1.0, cv::THRESH_BINARY);
        int kernel_size = (int) OPTDENSE::nPixelArea * 2 + 1;
        auto kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) / (float) (kernel_size * kernel_size);
        cv::filter2D(mask_img, depthData.confMap, mask_img.depth(), kernel);
    }

    // run gipuma to estimate depthMap and normalMap
    {
        int numberOfImages = depthData.images.GetSize();
        std::vector<cv::Mat_<float>> images(numberOfImages);
        std::vector<cv::Mat_<float>> projection_matrices(numberOfImages);
        for (int i = 0; i < numberOfImages; ++i) {
            images[i] = depthData.images[i].image;
            projection_matrices[i] = MatxtoMat(depthData.images[i].camera.P);
        }

        if (OPTDENSE::importReferenceDepth)
        {
            // init normal map
            int delta_x[] = {-1, 0, 1, 0};
            int delta_y[] = {0, 1, 0, -1};
            for (int y = 0 ; y < depthData.normalMap.rows; ++y) {
                for (int x = 0; x < depthData.normalMap.cols; ++x) {
                    Point3f ic(x, y, depthData.depthMap(y, x));
                    Point3f pc0 =
                            depthData.images.First().camera.TransformPointI2C(ic);
                    Point3f normal(0, 0, 0);
                    float weight = 0;
                    for (int k = 0; k < 4; ++k) {
                        int x1 = x + delta_x[k];
                        int y1 = y + delta_y[k];
                        int x2 = x + delta_x[(k + 1) % 4];
                        int y2 = y + delta_y[(k + 1) % 4];
                        if (depthData.depthMap.isInside({x1, y1}) && depthData.depthMap.isInside({x2, y2})) {
                            Point3f ic1(x1, y1, depthData.depthMap(y1, x1));
                            Point3f ic2(x2, y2, depthData.depthMap(y2, x2));
                            Point3f pc1 =
                                    depthData.images.First().camera.TransformPointI2C(ic1);
                            Point3f pc2 =
                                    depthData.images.First().camera.TransformPointI2C(ic2);

                            Point3f edge1(pc1-pc0);
                            Point3f edge2(pc2-pc0);
                            Point3f tmp_normal = edge1.cross(edge2);
                            weight += norm(tmp_normal);
                            normal += tmp_normal;
                        }
                    }
                    depthData.normalMap(y, x) = normalized(normal);
                }
            }
        }

#if TD_VERBOSE != TD_VERBOSE_OFF
        // save rough depth map as image
  if (g_nVerbosityLevel > 4) {
    depthData.depthMap.Save(ComposeDepthFilePath(idxImage, "init.pfm"));
    ExportDepthMap(ComposeDepthFilePath(idxImage, "init.png"), depthData.depthMap);
    ExportNormalMap(ComposeDepthFilePath(idxImage, "init.normal.png"), depthData.normalMap);
    ExportPointCloud(ComposeDepthFilePath(idxImage, "init.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
  }
#endif
        const String path(ComposeDepthFilePath(idxImage, "gipuma"));
        cv::Mat_<cv::Point3_<float>> normal_map = depthData.normalMap; // no deep copy
        gipuma::GipumaMain(path, images, projection_matrices, depthData.depthMap,
                           normal_map, depthData.dMin, depthData.dMax,
                           OPTDENSE::paramsFile.c_str(), OPTDENSE::importReferenceDepth);

#if 1 && TD_VERBOSE != TD_VERBOSE_OFF
        // save intermediate depth map as image
		if (g_nVerbosityLevel > 4) {
			depthData.depthMap.Save(path + ".depth.pfm");
      depthData.confMap.Save(path + ".conf.pfm");
			depthData.normalMap.Save(path + ".normal.pfm");
			ExportDepthMap(path + ".png", depthData.depthMap);
			ExportNormalMap(path + ".normal.png", depthData.normalMap);
			ExportPointCloud(path + ".ply", *depthData.images.First().pImageData, depthData.depthMap,
							 depthData.normalMap);
		}
#endif
    }

    return true;
}

bool DepthMapsData::CpuPatchMatch(IIndex idxImage, cList<SEACAVE::Thread> &threads, Image64F imageSum0, cList<DepthEstimator> &estimators)
{
    DepthData& depthData(arrDepthData[idxImage]);
    const DepthData::ViewData& image(depthData.images.First());
    const unsigned nMaxThreads(scene.nMaxThreads);
    volatile Thread::safe_t idxPixel;
    // run propagation and random refinement cycles on the reference data
	for (unsigned iter=0; iter<OPTDENSE::nEstimationIters; ++iter) {
		// create working threads
		const DepthEstimator::ENDIRECTION dir((DepthEstimator::ENDIRECTION)(iter%DepthEstimator::DIRS));
		idxPixel = -1;
		ASSERT(estimators.IsEmpty());
		while (estimators.GetSize() < nMaxThreads)
			estimators.AddConstruct(depthData, idxPixel, imageSum0, dir);
		ASSERT(estimators.GetSize() == threads.GetSize()+1);
		FOREACH(i, threads)
			threads[i].start(EstimateDepthMapTmp, &estimators[i]);
		EstimateDepthMapTmp(&estimators.Last());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.Release();
		#if 1 && TD_VERBOSE != TD_VERBOSE_OFF
		// save intermediate depth map as image
		if (g_nVerbosityLevel > 4) {
			const String path(ComposeDepthFilePath(image.pImageData-scene.images.Begin(), "iter")+String::ToString(iter));
			depthData.depthMap.Save(ComposeDepthFilePath(idxImage, "iter.depth.pfm"));
			depthData.normalMap.Save(ComposeDepthFilePath(idxImage, "iter.normal.pfm"));
			ExportDepthMap(path+".png", depthData.depthMap);
			ExportNormalMap(path+".normal.png", depthData.normalMap);
			ExportPointCloud(path+".ply", *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
		}
		#endif
	}

	return true;
}


// filter out small depth segments from the given depth map
bool DepthMapsData::RemoveSmallSegments(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*0.7f);
	unsigned speckle_size = OPTDENSE::nSpeckleSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	ImageRef size(depthMap.size());

	// allocate memory on heap for dynamic programming arrays
	TImage<bool> done_map(size, false);
	CAutoPtrArr<ImageRef> seg_list(new ImageRef[size.x*size.y]);
	unsigned seg_list_count;
	unsigned seg_list_curr;
	ImageRef neighbor[4];

	// for all pixels do
	for (int u=0; u<size.x; ++u) {
		for (int v=0; v<size.y; ++v) {
			// if the first pixel in this segment has been already processed => skip
			if (done_map(v,u))
				continue;

			// init segment list (add first element
			// and set it to be the next element to check)
			seg_list[0] = ImageRef(u,v);
			seg_list_count = 1;
			seg_list_curr  = 0;

			// add neighboring segments as long as there
			// are none-processed pixels in the seg_list;
			// none-processed means: seg_list_curr<seg_list_count
			while (seg_list_curr < seg_list_count) {
				// get address of current pixel in this segment
				const ImageRef addr_curr(seg_list[seg_list_curr]);

				// fill list with neighbor positions
				neighbor[0] = ImageRef(addr_curr.x-1, addr_curr.y  );
				neighbor[1] = ImageRef(addr_curr.x+1, addr_curr.y  );
				neighbor[2] = ImageRef(addr_curr.x  , addr_curr.y-1);
				neighbor[3] = ImageRef(addr_curr.x  , addr_curr.y+1);

				// for all neighbors do
				const Depth& depth_curr = depthMap(addr_curr);
				for (int i=0; i<4; ++i) {
					// get neighbor pixel address
					const ImageRef& addr_neighbor(neighbor[i]);
					// check if neighbor is inside image
					if (addr_neighbor.x>=0 && addr_neighbor.y>=0 && addr_neighbor.x<size.x && addr_neighbor.y<size.y) {
						// check if neighbor has not been added yet
						bool& done = done_map(addr_neighbor);
						if (!done) {
							// check if the neighbor is valid and similar to the current pixel
							// (belonging to the current segment)
							const Depth& depth_neighbor = depthMap(addr_neighbor);
							if (depth_neighbor>0 && IsDepthSimilar(depth_curr, depth_neighbor, fDepthDiffThreshold)) {
								// add neighbor coordinates to segment list
								seg_list[seg_list_count++] = addr_neighbor;
								// set neighbor pixel in done_map to "done"
								// (otherwise a pixel may be added 2 times to the list, as
								//  neighbor of one pixel and as neighbor of another pixel)
								done = true;
							}
						}
					}
				}

				// set current pixel in seg_list to "done"
				++seg_list_curr;

				// set current pixel in done_map to "done"
				done_map(addr_curr) = true;
			} // end: while (seg_list_curr < seg_list_count)

			// if segment NOT large enough => invalidate pixels
			if (seg_list_count < speckle_size) {
				// for all pixels in current segment invalidate pixels
				for (unsigned i=0; i<seg_list_count; ++i) {
					depthMap(seg_list[i]) = 0;
					if (!normalMap.empty()) normalMap(seg_list[i]) = Normal::ZERO;
					if (!confMap.empty()) confMap(seg_list[i]) = 0;
				}
			}
		}
	}

	return true;
} // RemoveSmallSegments
/*----------------------------------------------------------------*/

// try to fill small gaps in the depth map
bool DepthMapsData::GapInterpolation(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*2.5f);
	unsigned nIpolGapSize = OPTDENSE::nIpolGapSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	ImageRef size(depthMap.size());

	// 1. Row-wise:
	// for each row do
	for (int v=0; v<size.y; ++v) {
		// init counter
		unsigned count = 0;

		// for each element of the row do
		for (int u=0; u<size.x; ++u) {
			// get depth of this location
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if speckle is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)u > count) {
				// first value index for interpolation
				int u_curr(u-count);
				const int u_first(u_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap(v,u_first);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v,u_curr) = avg;
					} while (++u_curr<u);
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v,u_curr) = (d+=diff);
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v,u_first), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v,u_curr) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v,u_curr));
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);
					}
					#endif
				}
			}

			// reset counter
			count = 0;
		}
	}

	// 2. Column-wise:
	// for each column do
	for (int u=0; u<size.x; ++u) {

		// init counter
		unsigned count = 0;

		// for each element of the column do
		for (int v=0; v<size.y; ++v) {
			// get depth of this location
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if gap is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)v > count) {
				// first value index for interpolation
				int v_curr(v-count);
				const int v_first(v_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap(v_first,u);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v_curr,u) = avg;
					} while (++v_curr<v);
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v_first,u), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v_curr,u) = (d+=diff);
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v_first,u), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v_curr,u) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v_curr,u));
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);
					}
					#endif
				}
			}

			// reset counter
			count = 0;
		}
	}
	return true;
} // GapInterpolation
/*----------------------------------------------------------------*/


// filter depth-map, one pixel at a time, using confidence based fusion or neighbor pixels
bool DepthMapsData::FilterDepthMap(DepthData& depthDataRef, const IIndexArr& idxNeighbors, bool bAdjust)
{
	TD_TIMER_STARTD();

	// count valid neighbor depth-maps
	ASSERT(depthDataRef.IsValid() && !depthDataRef.IsEmpty());
	const IIndex N = idxNeighbors.GetSize();
	ASSERT(OPTDENSE::nMinViewsFilter > 0 && scene.nCalibratedImages > 1);
	const IIndex nMinViews(MINF(OPTDENSE::nMinViewsFilter,scene.nCalibratedImages-1));
	const IIndex nMinViewsAdjust(MINF(OPTDENSE::nMinViewsFilterAdjust,scene.nCalibratedImages-1));
	if (N < nMinViews || N < nMinViewsAdjust) {
		DEBUG("error: depth map %3u can not be filtered", depthDataRef.images.First().pImageData-scene.images.Begin());
		return false;
	}

	// project all neighbor depth-maps to this image
	const DepthData::ViewData& imageRef = depthDataRef.images.First();
	const Image8U::Size sizeRef(depthDataRef.depthMap.size());
	const Camera& cameraRef = imageRef.camera;
	DepthMapArr depthMaps(N);
	ConfidenceMapArr confMaps(N);
	FOREACH(n, depthMaps) {
		DepthMap& depthMap = depthMaps[n];
		depthMap.create(sizeRef);
		depthMap.memset(0);
		ConfidenceMap& confMap = confMaps[n];
		if (bAdjust) {
			confMap.create(sizeRef);
			confMap.memset(0);
		}
		const IIndex idxView = depthDataRef.neighbors[idxNeighbors[(IIndex)n]].idx.ID;
		const DepthData& depthData = arrDepthData[idxView];
		const Camera& camera = depthData.images.First().camera;
		const Image8U::Size size(depthData.depthMap.size());
		for (int i=0; i<size.height; ++i) {
			for (int j=0; j<size.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				ASSERT(depth > 0);
				const Point3 X(camera.TransformPointI2W(Point3(x.x,x.y,depth)));
				const Point3 camX(cameraRef.TransformPointW2C(X));
				if (camX.z <= 0)
					continue;
				#if 0
				// set depth on the rounded image projection only
				const ImageRef xRef(ROUND2INT(cameraRef.TransformPointC2I(camX)));
				if (!depthMap.isInside(xRef))
					continue;
				Depth& depthRef(depthMap(xRef));
				if (depthRef != 0 && depthRef < camX.z)
					continue;
				depthRef = camX.z;
				if (bAdjust)
					confMap(xRef) = depthData.confMap(x);
				#else
				// set depth on the 4 pixels around the image projection
				const Point2 imgX(cameraRef.TransformPointC2I(camX));
				const ImageRef xRefs[4] = {
					ImageRef(FLOOR2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(FLOOR2INT(imgX.x), CEIL2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), CEIL2INT(imgX.y))
				};
				for (int p=0; p<4; ++p) {
					const ImageRef& xRef = xRefs[p];
					if (!depthMap.isInside(xRef))
						continue;
					Depth& depthRef(depthMap(xRef));
					if (depthRef != 0 && depthRef < (Depth)camX.z)
						continue;
					depthRef = (Depth)camX.z;
					if (bAdjust)
						confMap(xRef) = depthData.confMap(x);
				}
				#endif
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (g_nVerbosityLevel > 3)
			ExportDepthMap(MAKE_PATH(String::FormatString("depthRender%04u.%04u.png", depthDataRef.images.First().pImageData-scene.images.Begin(), idxView)), depthMap);
		#endif
	}

	const float thDepthDiff(OPTDENSE::fDepthDiffThreshold*1.2f);
	DepthMap newDepthMap(sizeRef);
	ConfidenceMap newConfMap(sizeRef);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	size_t nProcessed(0), nDiscarded(0);
	#endif
	if (bAdjust) {
		// average similar depths, and decrease confidence if depths do not agree
		// (inspired by: "Real-Time Visibility-Based Fusion of Depth Maps", Merrell, 2007)
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// update best depth and confidence estimate with all estimates
				float posConf(depthDataRef.confMap(xRef)), negConf(0);
				Depth avgDepth(depth*posConf);
				unsigned nPosViews(0), nNegViews(0);
				unsigned n(N);
				do {
					const Depth d(depthMaps[--n](xRef));
					if (d == 0) {
						if (nPosViews + nNegViews + n < nMinViews)
							goto DiscardDepth;
						continue;
					}
					ASSERT(d > 0);
					if (IsDepthSimilar(depth, d, thDepthDiff)) {
						// average similar depths
						const float c(confMaps[n](xRef));
						avgDepth += d*c;
						posConf += c;
						++nPosViews;
					} else {
						// penalize confidence
						if (depth > d) {
							// occlusion
							negConf += confMaps[n](xRef);
						} else {
							// free-space violation
							const DepthData& depthData = arrDepthData[depthDataRef.neighbors[idxNeighbors[n]].idx.ID];
							const Camera& camera = depthData.images.First().camera;
							const Point3 X(cameraRef.TransformPointI2W(Point3(xRef.x,xRef.y,depth)));
							const ImageRef x(ROUND2INT(camera.TransformPointW2I(X)));
							if (depthData.confMap.isInside(x)) {
								const float c(depthData.confMap(x));
								negConf += (c > 0 ? c : confMaps[n](xRef));
							} else
								negConf += confMaps[n](xRef);
						}
						++nNegViews;
					}
				} while (n);
				ASSERT(nPosViews+nNegViews >= nMinViews);
				// if enough good views and positive confidence...
				if (nPosViews >= nMinViewsAdjust && posConf > negConf && ISINSIDE(avgDepth/=posConf, depthDataRef.dMin, depthDataRef.dMax)) {
					// consider this pixel an inlier
					newDepthMap(xRef) = avgDepth;
					newConfMap(xRef) = posConf - negConf;
				} else {
					// consider this pixel an outlier
					DiscardDepth:
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					#if TD_VERBOSE != TD_VERBOSE_OFF
					++nDiscarded;
					#endif
				}
			}
		}
	} else {
		// remove depth if it does not agree with enough neighbors
		const float thDepthDiffStrict(OPTDENSE::fDepthDiffThreshold*0.8f);
		const unsigned nMinGoodViewsProc(75), nMinGoodViewsDeltaProc(65);
		const unsigned nDeltas(4);
		const unsigned nMinViewsDelta(nMinViews*(nDeltas-2));
		const ImageRef xDs[nDeltas] = { ImageRef(-1,0), ImageRef(1,0), ImageRef(0,-1), ImageRef(0,1) };
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// check if very similar with the neighbors projected to this pixel
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					unsigned n(N);
					do {
						const Depth d(depthMaps[--n](xRef));
						if (d > 0) {
							// valid view
							++nViews;
							if (IsDepthSimilar(depth, d, thDepthDiffStrict)) {
								// agrees with this neighbor
								++nGoodViews;
							}
						}
					} while (n);
					if (nGoodViews < nMinViews || nGoodViews < nViews*nMinGoodViewsProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// check if similar with the neighbors projected around this pixel
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					for (unsigned d=0; d<nDeltas; ++d) {
						const ImageRef xDRef(xRef+xDs[d]);
						unsigned n(N);
						do {
							const Depth d(depthMaps[--n](xDRef));
							if (d > 0) {
								// valid view
								++nViews;
								if (IsDepthSimilar(depth, d, thDepthDiff)) {
									// agrees with this neighbor
									++nGoodViews;
								}
							}
						} while (n);
					}
					if (nGoodViews < nMinViewsDelta || nGoodViews < nViews*nMinGoodViewsDeltaProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// enough good views, keep it
				newDepthMap(xRef) = depth;
				newConfMap(xRef) = depthDataRef.confMap(xRef);
			}
		}
	}
	if (!SaveDepthMap(ComposeDepthFilePath(imageRef.pImageData-scene.images.Begin(), "filtered.dmap"), newDepthMap) ||
		!SaveConfidenceMap(ComposeDepthFilePath(imageRef.pImageData-scene.images.Begin(), "filtered.cmap"), newConfMap))
		return false;

	#if TD_VERBOSE != TD_VERBOSE_OFF
	DEBUG("Depth map %3u filtered using %u other images: %u/%u depths discarded (%s)", imageRef.pImageData-scene.images.Begin(), N, nDiscarded, nProcessed, TD_TIMER_GET_FMT().c_str());
	#endif

	return true;
} // FilterDepthMap
/*----------------------------------------------------------------*/

// fuse all valid depth-maps in the same 3D point cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
struct Proj {
	union {
		uint32_t idxPixel;
		struct {
			uint16_t x, y; // image pixel coordinates
		};
	};
	inline Proj() {}
	inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
	inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
	inline ImageRef GetCoord() const { return ImageRef(x,y); }
};
typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;
void DepthMapsData::FuseDepthMaps(PointCloud& pointcloud, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	// find best connected images
	IndexScoreArr connections(0, scene.images.GetSize());
	size_t nPointsEstimate(0);
	FOREACH(i, scene.images) {
		DepthData& depthData = arrDepthData[i];
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(i, "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		IndexScore& connection = connections.AddEmpty();
		connection.idx = i;
		connection.score = (float)scene.images[i].neighbors.GetSize();
		nPointsEstimate += ROUND2INT(depthData.depthMap.area()*(0.5f/*valid*/*0.3f/*new*/));
	}
	connections.Sort();

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, scene.images.GetSize()));
	CLISTDEF0(Depth*) invalidDepths(0, 32);
	size_t nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(scene.images.GetSize());
	ProjsArr projs(0, nPointsEstimate);
	pointcloud.points.Reserve(nPointsEstimate);
	pointcloud.pointViews.Reserve(nPointsEstimate);
	pointcloud.pointWeights.Reserve(nPointsEstimate);
	Util::Progress progress(_T("Fused depth-maps"), connections.GetSize());
	GET_LOGCONSOLE().Pause();
	FOREACHPTR(pConnection, connections) {
		TD_TIMER_STARTD();
		const uint32_t idxImage(pConnection->idx);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(!depthData.images.IsEmpty() && !depthData.neighbors.IsEmpty());
		for (const ViewScore& neighbor: depthData.neighbors) {
			const Image& imageData = scene.images[neighbor.idx.ID];
			DepthIndex& depthIdxs = arrDepthIdx[&imageData-scene.images.Begin()];
			if (depthIdxs.empty()) {
				depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
				depthIdxs.memset((uint8_t)NO_ID);
			}
		}
		ASSERT(!depthData.IsEmpty());
		const Image8U::Size sizeMap(depthData.depthMap.size());
		const Image& imageData = *depthData.images.First().pImageData;
		ASSERT(&imageData-scene.images.Begin() == idxImage);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const size_t nNumPointsPrev(pointcloud.points.GetSize());
		for (int i=0; i<sizeMap.height; ++i) {
			for (int j=0; j<sizeMap.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				++nDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = depthIdxs(x);
				if (idxPoint != NO_ID)
					continue;
				// create the corresponding 3D point
				idxPoint = (uint32_t)pointcloud.points.GetSize();
				PointCloud::Point& point = pointcloud.points.AddEmpty();
				point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth));
				PointCloud::ViewArr& views = pointcloud.pointViews.AddEmpty();
				views.Insert(idxImage);
				PointCloud::WeightArr& weights = pointcloud.pointWeights.AddEmpty();
				weights.Insert(depthData.confMap(x));
				ProjArr& pointProjs = projs.AddEmpty();
				pointProjs.Insert(Proj(x));
				// check the projection in the neighbor depth-maps
				REAL confidence(weights.First());
				Point3 X(point*confidence);
				invalidDepths.Empty();
				FOREACHPTR(pNeighbor, depthData.neighbors) {
					const IIndex idxImageB(pNeighbor->idx.ID);
					const Image& imageDataB = scene.images[idxImageB];
					const Point3f pt(imageDataB.camera.ProjectPointP3(point));
					if (pt.z <= 0)
						continue;
					const ImageRef xB(ROUND2INT(pt.x/pt.z), ROUND2INT(pt.y/pt.z));
					DepthData& depthDataB = arrDepthData[idxImageB];
					DepthMap& depthMapB = depthDataB.depthMap;
					if (!depthMapB.isInside(xB))
						continue;
					Depth& depthB = depthMapB(xB);
					if (depthB == 0)
						continue;
					uint32_t& idxPointB = arrDepthIdx[idxImageB](xB);
					if (idxPointB != NO_ID)
						continue;
					if (IsDepthSimilar(pt.z, depthB, OPTDENSE::fDepthDiffThreshold)) {
						// add view to the 3D point
						ASSERT(views.FindFirst(idxImageB) == PointCloud::ViewArr::NO_INDEX);
						const float confidenceB(depthDataB.confMap(xB));
						const IIndex idx(views.InsertSort(idxImageB));
						weights.InsertAt(idx, confidenceB);
						pointProjs.InsertAt(idx, Proj(xB));
						idxPointB = idxPoint;
						X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);
						confidence += confidenceB;
					} else
					if (pt.z < depthB) {
						// discard depth
						invalidDepths.Insert(&depthB);
					}
				}
				if (views.GetSize() < nMinViewsFuse) {
					// remove point
					FOREACH(v, views) {
						const IIndex idxImageB(views[v]);
						const ImageRef x(pointProjs[v].GetCoord());
						ASSERT(arrDepthIdx[idxImageB].isInside(x) && arrDepthIdx[idxImageB](x).idx != NO_ID);
						arrDepthIdx[idxImageB](x).idx = NO_ID;
					}
					projs.RemoveLast();
					pointcloud.pointWeights.RemoveLast();
					pointcloud.pointViews.RemoveLast();
					pointcloud.points.RemoveLast();
				} else {
					// this point is valid, store it
					point = X*(REAL(1)/confidence);
					ASSERT(ISFINITE(point));
					// invalidate all neighbor depths that do not agree with it
					for (Depth* pDepth: invalidDepths)
						*pDepth = 0;
				}
			}
		}
		ASSERT(pointcloud.points.GetSize() == pointcloud.pointViews.GetSize() && pointcloud.points.GetSize() == pointcloud.pointWeights.GetSize() && pointcloud.points.GetSize() == projs.GetSize());
		DEBUG_ULTIMATE("Depths map for reference image %3u fused using %u depths maps: %u new points (%s)", idxImage, depthData.images.GetSize()-1, pointcloud.points.GetSize()-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(pConnection-connections.Begin());
	}
	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();

	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%) (%s)", connections.GetSize(), nDepths, pointcloud.points.GetSize(), ROUND2INT((100.f*pointcloud.points.GetSize())/nDepths), TD_TIMER_GET_FMT().c_str());

	if (bEstimateNormal && !pointcloud.points.IsEmpty()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		TD_TIMER_STARTD();
		pointcloud.normals.Resize(pointcloud.points.GetSize());
		const int64_t nPoints((int64_t)pointcloud.points.GetSize());
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			PointCloud::WeightArr& weights = pointcloud.pointWeights[i];
			ASSERT(!weights.IsEmpty());
			IIndex idxView(0);
			float bestWeight = weights.First();
			for (IIndex idx=1; idx<weights.GetSize(); ++idx) {
				const PointCloud::Weight& weight = weights[idx];
				if (bestWeight < weight) {
					bestWeight = weight;
					idxView = idx;
				}
			}
			const DepthData& depthData(arrDepthData[pointcloud.pointViews[i][idxView]]);
			ASSERT(depthData.IsValid() && !depthData.IsEmpty());
			depthData.GetNormal(projs[i][idxView].GetCoord(), pointcloud.normals[i]);
		}
		DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.points.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// release all depth-maps
	FOREACHPTR(pDepthData, arrDepthData) {
		if (pDepthData->IsValid())
			pDepthData->DecRef();
	}
} // FuseDepthMaps
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

struct DenseDepthMapData {
	Scene& scene;
	IIndexArr images;
	IIndexArr neighborsMap;
	DepthMapsData detphMaps;
	volatile Thread::safe_t idxImage;
	SEACAVE::EventQueue events; // internal events queue (processed by the working threads)
	Semaphore sem;
	CAutoPtr<Util::Progress> progress;

	DenseDepthMapData(Scene& _scene)
		: scene(_scene), detphMaps(_scene), idxImage(0), sem(1) {}

	void SignalCompleteDepthmapFilter() {
		ASSERT(idxImage > 0);
		if (Thread::safeDec(idxImage) == 0)
			sem.Signal((unsigned)images.GetSize()*2);
	}
};

static void* DenseReconstructionEstimateTmp(void*);
static void* DenseReconstructionFilterTmp(void*);
bool Scene::DenseReconstruction()
{
	DenseDepthMapData data(*this);

	{
	// maps global view indices to our list of views to be processed
	IIndexArr imagesMap;

	// prepare images for dense reconstruction (load if needed)
	{
		TD_TIMER_START();
		data.images.Reserve(images.GetSize());
		imagesMap.Resize(images.GetSize());
		#ifdef DENSE_USE_OPENMP
		bool bAbort(false);
		#pragma omp parallel for shared(data, bAbort)
		for (int_t ID=0; ID<(int_t)images.GetSize(); ++ID) {
			#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const IIndex idxImage((IIndex)ID);
		#else
		FOREACH(idxImage, images) {
		#endif
			// skip invalid, uncalibrated or discarded images
			Image& imageData = images[idxImage];
			if (!imageData.IsValid()) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				imagesMap[idxImage] = NO_ID;
				continue;
			}
			// map image index
			#ifdef DENSE_USE_OPENMP
			#pragma omp critical
			#endif
			{
				imagesMap[idxImage] = data.images.GetSize();
				data.images.Insert(idxImage);
			}
			// reload image at the appropriate resolution
			const unsigned nMaxResolution(imageData.RecomputeMaxResolution(OPTDENSE::nResolutionLevel, OPTDENSE::nMinResolution));
			if (!imageData.ReloadImage(nMaxResolution)) {
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				continue;
				#else
				return false;
				#endif
			}
			imageData.UpdateCamera(platforms);
			// print image camera
			DEBUG_ULTIMATE("K%d = \n%s", idxImage, cvMat2String(imageData.camera.K).c_str());
			DEBUG_LEVEL(3, "R%d = \n%s", idxImage, cvMat2String(imageData.camera.R).c_str());
			DEBUG_LEVEL(3, "C%d = \n%s", idxImage, cvMat2String(imageData.camera.C).c_str());
		}
		#ifdef DENSE_USE_OPENMP
		if (bAbort || data.images.IsEmpty()) {
		#else
		if (data.images.IsEmpty()) {
		#endif
			VERBOSE("error: preparing images for dense reconstruction failed (errors loading images)");
			return false;
		}
		VERBOSE("Preparing images for dense reconstruction completed: %d images (%s)", images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// select images to be used for dense reconstruction
	{
		TD_TIMER_START();
		// for each image, find all useful neighbor views
		IIndexArr invalidIDs;
#ifdef DENSE_USE_OPENMP
#pragma omp parallel for shared(data, invalidIDs)
		for (int_t ID=0; ID<(int_t)data.images.GetSize(); ++ID) {
			const IIndex idx((IIndex)ID);
#else
  		FOREACH(idx, data.images) {
#endif
  			const IIndex idxImage(data.images[idx]);
  			ASSERT(imagesMap[idxImage] != NO_ID);
  			DepthData& depthData(data.detphMaps.arrDepthData[idxImage]);
  			if (!data.detphMaps.SelectViews(depthData)) {
#ifdef DENSE_USE_OPENMP
#pragma omp critical
#endif
  				invalidIDs.InsertSort(idx);
  			}
  		}
  		RFOREACH(i, invalidIDs) {
  			const IIndex idx(invalidIDs[i]);
  			imagesMap[data.images.Last()] = idx;
  			imagesMap[data.images[idx]] = NO_ID;
  			data.images.RemoveAt(idx);
  		}
  		// globally select a target view for each reference image
  		if (OPTDENSE::nNumViews == 1 && !data.detphMaps.SelectViews(data.images, imagesMap, data.neighborsMap)) {
  			VERBOSE("error: no valid images to be dense reconstructed");
  			return false;
  		}
  		ASSERT(!data.images.IsEmpty());
  		VERBOSE("Selecting images for dense reconstruction completed: %d images (%s)", data.images.GetSize(), TD_TIMER_GET_FMT().c_str());
  	}
	}

	// initialize the queue of images to be processed
	data.idxImage = 0;
	ASSERT(data.events.IsEmpty());
	data.events.AddEvent(new EVTProcessImage(0));
  if (OPTDENSE::algorithm == 0) {
    gipuma::selectCudaDevice();
  }
	// start working threads
	data.progress = new Util::Progress("Estimated depth-maps", data.images.GetSize());
	GET_LOGCONSOLE().Pause();
	if (nMaxThreads > 1) {
		// multi-thread execution
		cList<SEACAVE::Thread> threads(2);
		FOREACHPTR(pThread, threads)
			pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
		FOREACHPTR(pThread, threads)
			pThread->join();
	} else {
		// single-thread execution
		DenseReconstructionEstimate((void*)&data);
	}
	GET_LOGCONSOLE().Play();
	if (!data.events.IsEmpty())
		return false;
	data.progress.Release();

	if (exportDmapOnly)
		return true;

	if ((OPTDENSE::nOptimize & OPTDENSE::ADJUST_FILTER) != 0) {
		// initialize the queue of depth-maps to be filtered
		data.sem.Clear();
		data.idxImage = data.images.GetSize();
		ASSERT(data.events.IsEmpty());
		FOREACH(i, data.images)
			data.events.AddEvent(new EVTFilterDepthMap(i));
		// start working threads
		data.progress = new Util::Progress("Filtered depth-maps", data.images.GetSize());
		GET_LOGCONSOLE().Pause();
		if (nMaxThreads > 1) {
			// multi-thread execution
			cList<SEACAVE::Thread> threads(MINF(nMaxThreads, (unsigned)data.images.GetSize()));
			FOREACHPTR(pThread, threads)
				pThread->start(DenseReconstructionFilterTmp, (void*)&data);
			FOREACHPTR(pThread, threads)
				pThread->join();
		} else {
			// single-thread execution
			DenseReconstructionFilter((void*)&data);
		}
		GET_LOGCONSOLE().Play();
		if (!data.events.IsEmpty())
			return false;
		data.progress.Release();
	}

	// fuse all depth-maps
	pointcloud.Release();
	data.detphMaps.FuseDepthMaps(pointcloud, OPTDENSE::nEstimateNormals == 2);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (g_nVerbosityLevel > 2) {
		// print number of points with 3+ views
		size_t nPoints1m(0), nPoints2(0), nPoints3p(0);
		FOREACHPTR(pViews, pointcloud.pointViews) {
			switch (pViews->GetSize())
			{
			case 0:
			case 1:
				++nPoints1m;
				break;
			case 2:
				++nPoints2;
				break;
			default:
				++nPoints3p;
			}
		}
		VERBOSE("Dense point-cloud composed of:\n\t%u points with 1- views\n\t%u points with 2 views\n\t%u points with 3+ views", nPoints1m, nPoints2, nPoints3p);
	}
	#endif

	if (!pointcloud.IsEmpty()) {
		if (pointcloud.colors.IsEmpty() && OPTDENSE::nEstimateColors == 1)
			EstimatePointColors(images, pointcloud);
		if (pointcloud.normals.IsEmpty() && OPTDENSE::nEstimateNormals == 1)
			EstimatePointNormals(images, pointcloud);
	}
	return true;
} // DenseReconstructionDepthMap
/*----------------------------------------------------------------*/

void* DenseReconstructionEstimateTmp(void* arg) {
	const DenseDepthMapData& dataThreads = *((const DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionEstimate(arg);
	return NULL;
}

// initialize the dense reconstruction with the sparse point cloud
void Scene::DenseReconstructionEstimate(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	while (true) {
		CAutoPtr<Event> evt(data.events.GetEvent());
		switch (evt->GetID()) {
		case EVT_PROCESSIMAGE: {
			const EVTProcessImage& evtImage = *((EVTProcessImage*)(Event*)evt);
			if (evtImage.idxImage >= data.images.GetSize()) {
				if (nMaxThreads > 1) {
					// close working threads
					data.events.AddEvent(new EVTClose);
				}
				return;
			}
			// select views to reconstruct the depth-map for this image
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			// init images pair: reference image and the best neighbor view
			ASSERT(data.neighborsMap.IsEmpty() || data.neighborsMap[evtImage.idxImage] != NO_ID);
			if (!data.detphMaps.InitViews(depthData, data.neighborsMap.IsEmpty()?NO_ID:data.neighborsMap[evtImage.idxImage], OPTDENSE::nNumViews)) {
				// process next image
				data.events.AddEvent(new EVTProcessImage((IIndex)Thread::safeInc(data.idxImage)));
				break;
			}
			// try to load already compute depth-map for this image
			if (depthData.Load(ComposeDepthFilePath(idx, "dmap")) && !OPTDENSE::forceRecompute) {
				std::string depthFilePath = ComposeDepthFilePath(idx, "pfm");
				if (!exportDmapOnly && access(depthFilePath.c_str(), 0) != -1)
					depthData.depthMap.Load(depthFilePath);
				if (OPTDENSE::nOptimize & (OPTDENSE::OPTIMIZE)) {
					// optimize depth-map
					data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
				} else {
					// release image data
					depthData.ReleaseImages();
					depthData.Release();
				}
				// process next image
				data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			} else {
				data.events.AddEventFirst(new EVTEstimateDepthMap(evtImage.idxImage));
			}
			break;
		}
		case EVT_ESTIMATEDEPTHMAP: {
			const EVTEstimateDepthMap& evtImage = *((EVTEstimateDepthMap*)(Event*)evt);
			// request next image initialization to be performed while computing this depth-map
			data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			// extract depth map
			data.sem.Wait();
			if (OPTDENSE::importReferenceDepth) {
				const IIndex idx = data.images[evtImage.idxImage];
				DepthData& depthData(data.detphMaps.arrDepthData[idx]);
				data.detphMaps.InitDepthMapByRealsense(depthData, OPTDENSE::strRealsenseFileName, images[data.images[evtImage.idxImage]]);
			}
            data.detphMaps.EstimateDepthMap(data.images[evtImage.idxImage]);
			data.sem.Signal();
			if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
				// optimize depth-map
				data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
			} else {
				// save depth-map
				data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			}
			break;
		}
		case EVT_OPTIMIZEDEPTHMAP: {
			const EVTOptimizeDepthMap& evtImage = *((EVTOptimizeDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 3) {
				ExportDepthMap(ComposeDepthFilePath(idx, "raw.png"), depthData.depthMap);
				depthData.depthMap.Save(ComposeDepthFilePath(idx, "raw.pfm"));
			}
			#endif
			// apply filters
			if (OPTDENSE::nOptimize & (OPTDENSE::REMOVE_SPECKLES)) {
				TD_TIMER_START();
				data.detphMaps.RemoveSmallSegments(depthData);
				DEBUG_ULTIMATE("Depth-map %3u filtered: remove small segments (%s)", idx, TD_TIMER_GET_FMT().c_str());
			}
			if (OPTDENSE::nOptimize & (OPTDENSE::FILL_GAPS)) {
				TD_TIMER_START();
				data.detphMaps.GapInterpolation(depthData);
				DEBUG_ULTIMATE("Depth-map %3u filtered: gap interpolation (%s)", idx, TD_TIMER_GET_FMT().c_str());
			}
			// save depth-map
			data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			break;
		}
		case EVT_SAVEDEPTHMAP: {
			const EVTSaveDepthMap& evtImage = *((EVTSaveDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(idx, "png"), depthData.depthMap);
				ExportConfidenceMap(ComposeDepthFilePath(idx, "conf.png"), depthData.confMap);
				ExportPointCloud(ComposeDepthFilePath(idx, "ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
				if (g_nVerbosityLevel > 4) {
					ExportNormalMap(ComposeDepthFilePath(idx, "normal.png"), depthData.normalMap);
					depthData.confMap.Save(ComposeDepthFilePath(idx, "conf.pfm"));
				}
			}
			#endif
			// save compute depth-map for this image
			if (exportDmapOnly)
				depthData.depthMap.Save(ComposeDepthFilePath(idx, "raw.pfm"));
			depthData.Save(ComposeDepthFilePath(idx, "dmap"));
			depthData.ReleaseImages();
			depthData.Release();
			data.progress->operator++();
      break;
		}
		case EVT_CLOSE: {
			return;
		}
		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionEstimate
/*----------------------------------------------------------------*/

void* DenseReconstructionFilterTmp(void* arg) {
	DenseDepthMapData& dataThreads = *((DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionFilter(arg);
	return NULL;
}

// filter estimated depth-maps
void Scene::DenseReconstructionFilter(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	CAutoPtr<Event> evt;
	while ((evt=data.events.GetEvent(0)) != NULL) {
		switch (evt->GetID()) {
		case EVT_FILTERDEPTHMAP: {
			const EVTFilterDepthMap& evtImage = *((EVTFilterDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			if (!depthData.IsValid()) {
				data.SignalCompleteDepthmapFilter();
				break;
			}
			// make sure all depth-maps are loaded
			depthData.IncRef(ComposeDepthFilePath(idx, "dmap"));
			const unsigned numMaxNeighbors(8);
			IIndexArr idxNeighbors(0, depthData.neighbors.GetSize());
			FOREACH(n, depthData.neighbors) {
				const IIndex idxView = depthData.neighbors[n].idx.ID;
				DepthData& depthDataPair = data.detphMaps.arrDepthData[idxView];
				if (!depthDataPair.IsValid())
					continue;
				if (depthDataPair.IncRef(ComposeDepthFilePath(idxView, "dmap")) == 0) {
					// signal error and terminate
					data.events.AddEventFirst(new EVTFail);
					return;
				}
				idxNeighbors.Insert(n);
				if (idxNeighbors.GetSize() == numMaxNeighbors)
					break;
			}
			// filter the depth-map for this image
			if (data.detphMaps.FilterDepthMap(depthData, idxNeighbors, OPTDENSE::bFilterAdjust)) {
				// load the filtered maps after all depth-maps were filtered
				data.events.AddEvent(new EVTAdjustDepthMap(evtImage.idxImage));
			}
			// unload referenced depth-maps
			FOREACHPTR(pIdxNeighbor, idxNeighbors) {
				const IIndex idxView = depthData.neighbors[*pIdxNeighbor].idx.ID;
				DepthData& depthDataPair = data.detphMaps.arrDepthData[idxView];
				depthDataPair.DecRef();
			}
			depthData.DecRef();
			data.SignalCompleteDepthmapFilter();
			break; }

		case EVT_ADJUSTDEPTHMAP: {
			const EVTAdjustDepthMap& evtImage = *((EVTAdjustDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.detphMaps.arrDepthData[idx]);
			ASSERT(depthData.IsValid());
			data.sem.Wait();
			// load filtered maps
			if (depthData.IncRef(ComposeDepthFilePath(idx, "dmap")) == 0 ||
				!LoadDepthMap(ComposeDepthFilePath(idx, "filtered.dmap"), depthData.depthMap) ||
				!LoadConfidenceMap(ComposeDepthFilePath(idx, "filtered.cmap"), depthData.confMap))
			{
				// signal error and terminate
				data.events.AddEventFirst(new EVTFail);
				return;
			}
			ASSERT(depthData.GetRef() == 1);
			File::deleteFile(ComposeDepthFilePath(idx, "filtered.dmap").c_str());
			File::deleteFile(ComposeDepthFilePath(idx, "filtered.cmap").c_str());
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(idx, "filtered.png"), depthData.depthMap);
				ExportPointCloud(ComposeDepthFilePath(idx, "filtered.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
			// save filtered depth-map for this image
			depthData.Save(ComposeDepthFilePath(idx, "dmap"));
			depthData.DecRef();
			data.progress->operator++();
			break; }

		case EVT_FAIL: {
			data.events.AddEventFirst(new EVTFail);
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionFilter

/*----------------------------------------------------------------*/
