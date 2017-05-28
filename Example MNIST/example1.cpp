#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "CNN/cnn.h"

using namespace std;

float train( vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected )
{
	for ( int i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 )
			activate( layers[i], data );
		else
			activate( layers[i], layers[i - 1]->out );
	}

	tensor_t<float> grads = layers.back()->out - expected;

	for ( int i = layers.size() - 1; i >= 0; i-- )
	{
		if ( i == layers.size() - 1 )
			calc_grads( layers[i], grads );
		else
			calc_grads( layers[i], layers[i + 1]->grads_in );
	}

	for ( int i = 0; i < layers.size(); i++ )
	{
		fix_weights( layers[i] );
	}

	float err = 0;
	for ( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ )
	{
		float f = expected.data[i];
		if ( f > 0.5 )
			err += abs(grads.data[i]);
	}
	return err * 100;
}


void forward( vector<layer_t*>& layers, tensor_t<float>& data )
{
	for ( int i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 )
			activate( layers[i], data );
		else
			activate( layers[i], layers[i - 1]->out );
	}
}

struct case_t
{
	tensor_t<float> data;
	tensor_t<float> out;
};

uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
	streamsize size = file.tellg();
	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	return buffer;
}

vector<case_t> read_test_cases()
{
	vector<case_t> cases;

	uint8_t* train_image = read_file( "train-images.idx3-ubyte" );
	uint8_t* train_labels = read_file( "train-labels.idx1-ubyte" );

	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for ( int i = 0; i < case_count; i++ )
	{
		case_t c {tensor_t<float>( 28, 28, 1 ), tensor_t<float>( 10, 1, 1 )};

		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;

		for ( int x = 0; x < 28; x++ )
			for ( int y = 0; y < 28; y++ )
				c.data( x, y, 0 ) = img[x + y * 28] / 255.f;

		for ( int b = 0; b < 10; b++ )
			c.out( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

		cases.push_back( c );
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}

int main()
{
	vector<case_t> cases = read_test_cases();

	vector<layer_t*> layers;

	conv_layer_t * layer1 = new conv_layer_t( 1, 5, 8, cases[0].data.size );		// 28 * 28 * 1 -> 24 * 24 * 8
	relu_layer_t * layer2 = new relu_layer_t( layer1->out.size );
	pool_layer_t * layer3 = new pool_layer_t( 2, 2, layer2->out.size );				// 24 * 24 * 8 -> 12 * 12 * 8
	fc_layer_t * layer4 = new fc_layer_t(layer3->out.size, 10);					// 4 * 4 * 16 -> 10

	layers.push_back( (layer_t*)layer1 );
	layers.push_back( (layer_t*)layer2 );
	layers.push_back( (layer_t*)layer3 );
	layers.push_back( (layer_t*)layer4 );



	float amse = 0;
	int ic = 0;

	for ( long ep = 0; ep < 100000; )
	{

		for ( case_t& t : cases )
		{
			float xerr = train( layers, t.data, t.out );
			amse += xerr;

			ep++;
			ic++;

			if ( ep % 1000 == 0 )
				cout << "case " << ep << " err=" << amse/ic << endl;

			// if ( GetAsyncKeyState( VK_F1 ) & 0x8000 )
			// {
			//	   printf( "err=%.4f%\n", amse / ic  );
			//	   goto end;
			// }
		}
	}
	// end:



	while ( true )
	{
		uint8_t * data = read_file( "test.ppm" );

		if ( data )
		{
			uint8_t * usable = data;

			while ( *(uint32_t*)usable != 0x0A353532 )
				usable++;

#pragma pack(push, 1)
			struct RGB
			{
				uint8_t r, g, b;
			};
#pragma pack(pop)

			RGB * rgb = (RGB*)usable;

			tensor_t<float> image(28, 28, 1);
			for ( int i = 0; i < 28; i++ )
			{
				for ( int j = 0; j < 28; j++ )
				{
					RGB rgb_ij = rgb[i * 28 + j];
					image( j, i, 0 ) = (((float)rgb_ij.r
							     + rgb_ij.g
							     + rgb_ij.b)
							    / (3.0f*255.f));
				}
			}

			forward( layers, image );
			tensor_t<float>& out = layers.back()->out;
			for ( int i = 0; i < 10; i++ )
			{
				printf( "[%i] %f\n", i, out( i, 0, 0 )*100.0f );
			}

			delete[] data;
		}

		struct timespec wait;
		wait.tv_sec = 1;
		wait.tv_nsec = 0;
		nanosleep(&wait, nullptr);
	}
	return 0;
}
