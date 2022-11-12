# Configure Magnum build

# Corrade
add_subdirectory(external/corrade EXCLUDE_FROM_ALL)

# Magnum
set(MAGNUM_WITH_GLFWAPPLICATION ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_MESHTOOLS ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_PRIMITIVES ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_SHADERS ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_DEBUGTOOLS ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_ANYIMAGEIMPORTER ON CACHE BOOL "" FORCE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/external/magnum-bootstrap/modules")
add_subdirectory(external/magnum EXCLUDE_FROM_ALL)

# Plugins
set(MAGNUM_WITH_STBIMAGEIMPORTER ON CACHE BOOL "" FORCE)
set(BUILD_PLUGINS_STATIC ON CACHE BOOL "" FORCE)
add_subdirectory(external/magnum-plugins EXCLUDE_FROM_ALL)

# Integration (ImGui)
set(MAGNUM_WITH_IMGUI ON CACHE BOOL "" FORCE)
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/external/imgui")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/external/magnum-integration/modules")
add_subdirectory(external/magnum-integration EXCLUDE_FROM_ALL)

# ImPlot
#include_directories(external/implot)
add_library(ImPlot
	external/implot/implot.cpp
	external/implot/implot_items.cpp)
target_include_directories(ImPlot INTERFACE external/implot)
target_include_directories(ImPlot PRIVATE external/imgui)


# Remove unwanted hierarchy in the build directory introduced by Magnum
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
