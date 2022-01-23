# Configure Magnum build

# Corrade
add_subdirectory(external/corrade EXCLUDE_FROM_ALL)

# Magnum
set(WITH_GLFWAPPLICATION ON CACHE BOOL "" FORCE)
set(WITH_MESHTOOLS ON CACHE BOOL "" FORCE)
set(WITH_PRIMITIVES ON CACHE BOOL "" FORCE)
set(WITH_SHADERS ON CACHE BOOL "" FORCE)
set(WITH_DEBUGTOOLS ON CACHE BOOL "" FORCE)
set(WITH_ANYIMAGEIMPORTER ON CACHE BOOL "" FORCE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/external/magnum-bootstrap/modules")
add_subdirectory(external/magnum EXCLUDE_FROM_ALL)

# Plugins
set(WITH_STBIMAGEIMPORTER ON CACHE BOOL "" FORCE)
set(BUILD_PLUGINS_STATIC ON CACHE BOOL "" FORCE)
add_subdirectory(external/magnum-plugins EXCLUDE_FROM_ALL)

# Integration
set(WITH_IMGUI ON CACHE BOOL "" FORCE)
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/external/imgui")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/external/magnum-integration/modules")
add_subdirectory(external/magnum-integration EXCLUDE_FROM_ALL)

# Remove unwanted hierarchy in the build directory introduced by Magnum
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
