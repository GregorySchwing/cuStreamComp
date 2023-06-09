# cuStreamComp
Efficient CUDA Stream Compaction Library

Based on the folllowing works:

1. Markus Billeter et al. Efficient Stream Compaction on Wide SIMD Many-Core Architectures

2. InK-Compact-: In kernel Stream Compaction and Its Application to Multi-kernel Data Visualization on GPGPU- D.M. Hughes

3. Darius Bakunas-Milanowski et al. Efficient Algorithms for Stream Compaction on GPUs

It is an CUDA efficient implementation of the stream compaction algorithm based on **warp ballotting intrinsic**.

# How to use it
Its usage is straightforward:

 - Create a predicate functor to decide whether an element is valid or not.
```
struct predicate
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x>0;
	}
};
```

- Call the compact procedure to obtain the compacted array `d_output`.

```
cuCompactor::compact<int>(d_data,d_output,length,predicate(),blockSize);
cuCompactor::compactHybrid<int>(d_data,d_output,length,predicate(),blockSize);
cuCompactor::compactThrust<int>(d_data,d_output,length,predicate());

```

Note that both the input `d_data` and the output  `d_output` arrays have to be allocated on device.


*PERFORMANCE*

![Picture1](https://github.com/GregorySchwing/cuStreamComp/assets/39970712/11365c93-df01-474e-bd71-e95c27bb91be)
Thrust (T) performs both the Billeter (B) and Bakunas-Milanowski (H) implementations for all streams of size up to 1024*2^19.
1024*2^20 >= are too large for thrust to compact on my Quadro RTX 5000 Mobile / Max-Q with 16 GiBytes of VRAM.
B and H are more memory frugal that thrust and can perform compaction on a stream of at least 1024*2^20 elements.
