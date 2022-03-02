#pragma once

#include <Corrade/Utility/Resource.h>

struct ShaderSources
{
	static const ShaderSources& instance();
	ShaderSources() { CORRADE_RESOURCE_INITIALIZE(ShaderResources); }
	~ShaderSources() { CORRADE_RESOURCE_FINALIZE(ShaderResources); }
};
