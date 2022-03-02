#include <ShaderSources.hpp>

const ShaderSources& ShaderSources::instance()
{
	static const ShaderSources shaderSources;
	return shaderSources;
}
