#pragma once


struct gradient_t
{
	float grad;
	float oldgrad;
	gradient_t()
	{
		grad = 0;
		oldgrad = 0;
	}
};