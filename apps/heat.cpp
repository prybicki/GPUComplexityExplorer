#include <graphics/Visualizer.hpp>
#include <core/ResourceManager.hpp>
#include <macros/todo.hpp>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <compute/kernels.hpp>



#include <unistd.h>
#include <data/data.hpp>

int main(int argc, char** argv)
{
	Visualizer v {argc, argv};
	Field2D heatField {
		NCube2f::centeredAt({0.0f, 0.0f}, {1024.0f, 1024.0f}),
		Vec2i {128, 128},
		PropertySet {
			Property {f32, "heat", true}
		}
	};

	auto heatFieldArea = heatField.getArea();
	v.cameraLookAtWithPadding(heatFieldArea, 0.1f);

	auto square = heatFieldArea.scaled(0.1f);
	auto rectVertical = heatFieldArea.scaled({0.1f, 0.2f});
	auto rectHorizontal = heatFieldArea.scaled({0.2f, 0.1f});
	auto p2 = square.placedIn(heatFieldArea, {0.5f, 0.5f});
	auto p1 = square.placedIn(heatFieldArea, {0.1f, 0.1f});
	auto p3 = square.placedIn(heatFieldArea, {0.9f, 0.9f});
	auto p4 = rectVertical.placedIn(heatFieldArea, {0.33f, 0.66f});
	auto p5 = rectHorizontal.placedIn(heatFieldArea, {0.66f, 0.33f});

	fmt::print("{}\n{}\n{}\n{}\n{}\n", p1, p2, p3, p4, p5);

	// Continue here
	// Make sure everything is correct

	// NCube2i rect1 = {.min={1 * sizeX/4, 1 * sizeY/4}, .max={1 * sizeX/4 + 1 * 42, 1 * sizeY/4 + 1 * 42}};
	// NCube2i rect2 = {.min={2 * sizeX/4, 2 * sizeY/4}, .max={2 * sizeX/4 + 2 * 42, 2 * sizeY/4 + 2 * 42}};
	// NCube2i rect3 = {.min={3 * sizeX/4, 3 * sizeY/4}, .max={3 * sizeX/4 + 3 * 42, 3 * sizeY/4 + 3 * 42}};
    //
	// NCube2i rect4 = {.min={1 * sizeX/4, 2 * sizeY/4}, .max={1 * sizeX/4 + 2 * (sizeX/16), 2 * sizeY/4 + 6 * (sizeY/16)}};
	// NCube2i rect5 = {.min={2 * sizeX/4, 1 * sizeY/4}, .max={2 * sizeX/4 + 6 * (sizeX/16), 1 * sizeY/4 + 2 * (sizeY/16)}};
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect1, 1.0f);
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect2, 0.9f);
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect3, 0.8f);
    //
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect4, 0.8f);
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect5, 0.8f);
    //
	// while (v.mainLoopIteration())
	// {
	// 	auto begin = thrust::device_ptr<float>(fieldCurr);
	// 	auto end = thrust::device_ptr<float>(fieldCurr + sizeX * sizeY);
	// 	thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> minMaxPtrs = thrust::minmax_element(begin, end);
	// 	rm.run({sizeX * sizeY}, kTmpColorizeCustomF32, sizeX * sizeY, fieldCurr, minMaxPtrs.first.get(), minMaxPtrs.second.get(), fieldCol);
	// 	v.renderTexture(0, 0, sizeX, sizeY, fieldCol);
	// 	v.redraw();
    //
	// 	for (int i = 0; i < 100; i++) {
	// 		rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kHeatTransfer, sizeX, sizeY, fieldCurr, fieldNext, 0.01f);
	// 		std::swap(fieldCurr, fieldNext);
    //
	// 	}
	// }
}
