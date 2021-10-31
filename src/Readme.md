# Bert optimization based on graph compiler

## Adopted optimization

* Relay
In the optimization task, we will focus on hardware-independent optimization based on passes in TVM relay, including constant folding, dead-code elimination and other passes on tensor calculation like transformation, scaling factor folding, etc. 
	* Adopted passes on different levels
		- Module Level Pass
		- Function Level Pass
		- Sequential Level Pass

	* Customized self-defined pass
	Due to a default of EliminateCommonSubexpr with reshape operator, the EliminateCommonSubexpr is customized with python decorator.

* Auto-Scheduler (AutoTVM or Ansor)
Except for handicraft optimizations by pass, TVM offers the possibility to tune the scheduler automatically. Since previous paper already told me that AutoTVM might even worsen the performance if we use CPU, here I will use AutoTVM and Ansor which claims to have better performance.

* Tir
Little effort was done on Tir Transform, not to say on LLVM or CUDA, but it will be super interesting to work on that direction.

* DAG visualization & benchmarking

* Config
The benchmark I have done was totally on GCP CPU instance due to a lack of ressource, and the demand of GPU will take two days for processing, but I strongly recommend to test it on GPU.

My config:
GCP n1-standard-8( 8 Intel Skylake vCPU, 30 GB memory)
System: Ubuntu 16.04 LTS

* BERT tasks
Here we focus on the inference tasks

## Code structure


