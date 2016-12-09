#pragma once
#include "gradient_t.h"

#define LEARNING_RATE 0.01
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

static float update_weight( float w, gradient_t& grad, float multp = 1 )
{
	float m = (grad.grad + grad.oldgrad * MOMENTUM);
	w -= LEARNING_RATE  * m * multp +
		 LEARNING_RATE * WEIGHT_DECAY * w;
	return w;
}

static void update_gradient( gradient_t& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}