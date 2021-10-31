# Bert optimization based on graph compiler

## Adopted optimization

* Relay
In the optimization task, we will focus on hardware-independent optimization based on passes in TVM relay, including constant folding, dead-code elimination and special passes on tensor calculation like transformation, scaling factor folding, etc. 
	* Adopted passes on different levels
		- Module Level Pass
		- Function Level Pass
		- Sequential Level Pass

	* Customized self-defined pass (Chapter 8)
	`TVM_REGISTER_GLOBAL` 

* Tir
Little effort was done on Tir Transform, not to say on LLVM or CUDA, but it will be super interesting to work on that direction.

* Auto-Scheduler (AutoTVM or Ansor)
Except for handicraft optimizations, TVM offers the possibility to tune the scheduler automatically. Since previous paper already told me that AutoTVM might even worsen the performance if we use CPU, I will try to implement Ansor which claim to have better performance.
(Low-level compute logic depends on hardware)

* DAG visualization & benchmarking

* Config
The benchmark I have done was totally on CPU due to a lack of ressource, but I strongly recommend you to test it on GPU.

My config:
GCP n1-standard-8( 8 Intel Skylake vCPU, 30 GB memory)

* BERT tasks
Here we focus on the inference tasks

## Code structure
