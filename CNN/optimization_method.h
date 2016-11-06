#pragma once
#include "gradient_t.h"

#define LEARNING_RATE 0.01
#define MOMENTUM 0.4

static float update_weight( float w, gradient_t& grad, float multp = 1 )
{
	float m = (grad.grad + grad.oldgrad * MOMENTUM);
	w -= LEARNING_RATE  * m * multp;
	return w;
}

static void update_gradient( gradient_t& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}