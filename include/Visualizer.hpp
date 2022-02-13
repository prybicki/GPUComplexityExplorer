#pragma once

#include <cassert>
#include <functional>
#include <set>
#include <queue>

#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/MatrixTransformation2D.hpp>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/DebugTools/FrameProfiler.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Corrade/Utility/Arguments.h>

#include <implot.h>

#include <Vector.hpp>
#include <Field2DShader.hpp>

struct Visualizer : public Magnum::Platform::Application
{
	// TODO: Add documentation (units!)
	constexpr static float INITIAL_RENDER_DISTANCE = 16.0f;
	constexpr static float CAMERA_PAN_SPEED = 0.016f;
	constexpr static float CAMERA_ZOOM_SPEED = 0.032f;
	constexpr static Magnum::Vector2 INITAL_CAMERA_POSITION = {2048.f, 2048.f};

public:
	Visualizer(int argc, char** argv) : Visualizer({argc, argv}, makeCLIArgs({argc, argv})) {}
	~Visualizer();

	void setUserGUI(std::function<void()> userGUI);
	void renderParticles(count_t count, Vec2f* dPosition, float* dRadius, Vec4f* dColor);
	void renderTexture(float posX, float posY, int sizeX, int sizeY, void* devBytes);

private:
	Visualizer(const Arguments& args, const Corrade::Utility::Arguments& cliArgs);

	void handleKeyboard();
	void viewportEvent(ViewportEvent& event) override;
	void keyPressEvent(KeyEvent& event) override;
	void keyReleaseEvent(KeyEvent& event) override;
	void mousePressEvent(MouseEvent& event) override;
	void mouseReleaseEvent(MouseEvent& event) override;
	void mouseMoveEvent(MouseMoveEvent& event) override;
	void mouseScrollEvent(MouseScrollEvent& event) override;

	void drawGUI();
	void drawEvent() override;
	void updateProjectionMatrix();

	static Configuration makeWindowConfig(const Magnum::Utility::Arguments& cliArgs);
	static GLConfiguration makeOpenGLConfig();
	static Magnum::Utility::Arguments makeCLIArgs(const Arguments& args);

private:
	Magnum::Utility::Arguments cliArgs;
	Magnum::GL::Mesh circle;
	Magnum::Shaders::FlatGL2D flatShader;
	Field2DShader field2DShader;
	std::queue<std::function<void()>> drawQueue;

	std::set<KeyEvent::Key> pressedKeys;
	Magnum::SceneGraph::Object<Magnum::SceneGraph::MatrixTransformation2D> cameraObject;
	Magnum::SceneGraph::Camera2D camera;
	float currentZoom = 1.0f;

	Magnum::GL::Buffer colorBuffer;
	Magnum::GL::Buffer transformBuffer;
	cudaGraphicsResource_t colorResource;
	cudaGraphicsResource_t transformResource;

	std::optional<std::function<void()>> userGUI;
	Magnum::ImGuiIntegration::Context _imgui{Magnum::NoCreate};
	ImPlotContext* implotCtx;
	Magnum::DebugTools::FrameProfilerGL profiler { Magnum::DebugTools::FrameProfilerGL::Value::FrameTime, 10};
};
