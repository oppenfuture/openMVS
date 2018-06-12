#include <iostream>
#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"

using namespace MVS;

#define APPNAME _T("ReplaceMesh")

int main(int argc, LPCTSTR* argv) {
  if (argc < 4) {
    std::cerr << "Usage: ReplaceMesh working_folder input_mvs output_mvs" << std::endl;
    return 1;
  }

  WORKING_FOLDER = argv[1];
  INIT_WORKING_FOLDER;

  Scene scene(0);
  if (!scene.Load(MAKE_PATH_SAFE(String(argv[2])))) {
    std::cerr << "Failed to load " << argv[2] << std::endl;
    return 1;
  }

  std::cout << "Input new mesh name" << std::endl;
  std::string filename;
  std::cin >> filename;

  if (!scene.mesh.Load(MAKE_PATH_SAFE(filename))) {
    std::cerr << "Failed to load " << filename << std::endl;
    return 1;
  }

  if (!scene.Save(MAKE_PATH_SAFE(String(argv[3])))) {
    std::cerr << "Failed to save " << argv[3] << std::endl;
    return 1;
  }

  return 0;
}
