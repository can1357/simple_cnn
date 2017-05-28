#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct dropout_layer_t
{
	layer_type type = layer_type::dropout_layer;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<bool> hitmap;
	float p_activation;

	dropout_layer_t( tdsize in_size, float p_activation )
		:
		in( in_size.x, in_size.y, in_size.z ),
		out( in_size.x, in_size.y, in_size.z ),
		hitmap( in_size.x, in_size.y, in_size.z ),
		grads_in( in_size.x, in_size.y, in_size.z ),
		p_activation( p_activation )
	{
		
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		for ( int i = 0; i < in.size.x*in.size.y*in.size.z; i++ )
		{
			bool active = (rand() % RAND_MAX) / float( RAND_MAX ) <= p_activation;
			hitmap.data[i] = active;
			out.data[i] = active ? in.data[i] : 0.0f;
		}
	}


	void fix_weights()
	{
		
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int i = 0; i < in.size.x*in.size.y*in.size.z; i++ )
			grads_in.data[i] = hitmap.data[i] ? grad_next_layer.data[i] : 0.0f;
	}
};
#pragma pack(pop)