#include <random>

#include <fmt/format.h>

#include <Visualizer.hpp>
#include <Vector.hpp>
#include <cuda/kernels.hpp>

std::default_random_engine engine;

struct Particles
{
	count_t count;
	float* radius;
	Vec2f* position;
	Vec2f* velocity;
	Vec4f* color;

	Particles(count_t count, Vec2f posOrigin, float posRad, float radMin, float radMax, Vec4f col) : count(count)
	{
		std::vector<float> hRad;
		std::vector<Vec2f> hPos, hVel;
		std::vector<Vec4f> hCol;

		std::uniform_real_distribution<float> dRad {radMin, radMax};
		for (count_t i = 0; i < count; ++i) {

			hPos.emplace_back(posOrigin + randomVectorInRing(0.0f, posRad));
			hVel.emplace_back(randomVectorInRing(0.1f, 1.0f));
			hCol.emplace_back(col);
			hRad.emplace_back(dRad(engine));
		}
		CHECK_CUDA(cudaMalloc(&position, sizeof(Vec2f) * count));
		CHECK_CUDA(cudaMalloc(&velocity, sizeof(Vec2f) * count));
		CHECK_CUDA(cudaMalloc(&color, sizeof(Vec4f) * count));
		CHECK_CUDA(cudaMalloc(&radius, sizeof(float) * count));

		CHECK_CUDA(cudaMemcpy(position, hPos.data(), sizeof(Vec2f) * count, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(velocity, hVel.data(), sizeof(Vec2f) * count, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(color, hCol.data(), sizeof(Vec4f) * count, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(radius, hRad.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
	}

private:
	Vec2f randomVectorInRing(float r, float R)
	{
		std::uniform_real_distribution<float> dAngle {0, 2 * M_PI};
		std::uniform_real_distribution<float> dLen {r, R};

		float angle = dAngle(engine);
		return dLen(engine) * Vec2f(sinf(angle), cosf(angle));
	}
};

using namespace std::literals;

int main(int argc, char** argv)
{
	constexpr count_t count = 16384 * 1;
	Visualizer vis(argc, argv);

	std::vector<Vec4f> colors = {
		{1.0f, 0.0f, 0.0f, 0.8f},
		{1.0f, 1.0f, 0.0f, 0.8f},
		{0.0f, 1.0f, 0.0f, 0.8f},

		{0.0f, 1.0f, 1.0f, 0.8f},
		{0.0f, 0.0f, 1.0f, 0.8f},
		{1.0f, 0.0f, 1.0f, 0.8f},

		{1.0f, 1.0f, 1.0f, 0.8f},
		{0.5f, 0.5f, 0.5f, 0.8f},
		{0.0f, 0.0f, 0.0f, 1.0f}
	};

	// Fill background to avoid the ugly.
	vis.redraw();
	vis.mainLoopIteration();

	std::vector<Particles> particleGroups;
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			auto pg = Particles{count, {static_cast<float>(x), static_cast<float>(y)}, 0.1f, 0.001, 0.002, colors.at(3 * y + x)};
			particleGroups.push_back(pg);
		}
	}

	float dt = 0.0001f;
	vis.setUserGUI([&](){
		ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::SliderFloat( "DT", &dt, 0, 0.01, "%.4g", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
		ImGui::End();
	});

	bool shouldContinue;
	do {
		for (auto&& pg : particleGroups){
			cm.runSync1D(count, 256, kApplyVelocity, count, dt, pg.velocity, pg.position);
			vis.renderParticles(count, pg.position, pg.radius, pg.color);
		}
		vis.redraw();
		shouldContinue = vis.mainLoopIteration();
	}
	while (shouldContinue);
	return 0;
}
