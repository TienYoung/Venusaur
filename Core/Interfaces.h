#pragma once

namespace Venusaur
{
	class IDrawable
	{
	public:
		virtual void Draw() = 0;
		virtual const char* GetName() const = 0;
	};
}