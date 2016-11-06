# MNIST Examples

This is an example project based on simple_cnn for handwritten digit recognition using MNIST database.

##Example Topology 1:
```
conv_layer_t( 1, 5, 8, ... );			// 28 * 28 * 1 -> 24 * 24 * 8
relu_layer_t( ... );               
pool_layer_t( 2, 2, ... );				// 24 * 24 * 8 -> 12 * 12 * 8
fc_layer_t( ..., 10 );					// 4 * 4 * 16 -> 10
```

###Results (Youtube Video):

[![Youtube Video](https://img.youtube.com/vi/0nsxYBDTojs/0.jpg)](https://www.youtube.com/watch?v=0nsxYBDTojs)



##Example Topology 2:

```
conv_layer_t( 1, 5, 8, ... );			// 28 * 28 * 1 -> 24 * 24 * 8
relu_layer_t( ... );               
pool_layer_t( 2, 2, ... );				// 24 * 24 * 8 -> 12 * 12 * 8
conv_layer_t( 1, 3, 10, ... );			// 12 * 12 * 6 -> 10 * 10 * 10
relu_layer_t( ... );               
pool_layer_t( 2, 2, ... );				// 10 * 10 * 10 -> 5 * 5 * 10
fc_layer_t( ..., 10 );					// 4 * 4 * 16 -> 10
```

###Results (Youtube Video):

[![Youtube Video](https://img.youtube.com/vi/afLUb6lFTCk/0.jpg)](https://www.youtube.com/watch?v=afLUb6lFTCk)


