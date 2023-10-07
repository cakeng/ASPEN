## NeurIPS'23 Submission Build of ASPEN: Breaking Operator Barriers for Efficient Parallelization of Deep Neural Networks

This repo provides the source code for the proof-of-concept CPU build of the ASPEN system used in the NeurIPS'23 submission.

We also included three simple usage examples of the ASPEN system, which was provided with the supplimentary material of the NeurIPS'23 submission.

The "src" directory includes the source code of ASPEN. 

The "include" directory includes the header files of ASPEN. 

The "files" directory includes the miscellanous files for the examples.

The "examples" directory includes three examples of ASPEN as follows:

1. Executing ResNet-50 Inference using ASPEN.
2. Executing batched muti-DNN co-inference of ResNet-50 and VGG-16 using ASPEN.
3. Migrating and executing a custom DNN from PyTorch to ASPEN.

The detailed instruction for each example is included in each directory, in the "instructions.txt" file.

Run "make" to compile the libaspen.a library. The examples are built step by step, as described in the instructions of each example.

The given Makefile is configured to compile the ASPEN library with an AVX2 backend.

---

ASPEN has a dependency on OpenMP. The examples of this supplementary material have dependencies on PyTorch, TorchVision, and GCC.
