# Advance Deep Learning Labs Report

## Lab 1
### Questions:

![Questions_1](https://github.com/user-attachments/assets/94784592-a155-4b27-9c9e-40dfbbe4fc1f)

#### 1A
![Lab1_1_A](https://github.com/user-attachments/assets/92947434-fc5d-4aa5-a22b-3fb3e20ec7db)

Based on the plot, we can see the accuracy stagnate 


#### 1B
#### 2A
#### 2B



## Lab 2
## Lab 3

## Lab 4
### Questions:
![Questions_4](https://github.com/user-attachments/assets/e1402b90-ebeb-4ef5-8f93-83a4f8daad96)

### 1A
There are many factors that can impact the run-time of torch.compile. Some of the factors included could be the size of the model, 
the batch size, the amount of operations executed and the CPU's ability. One way to maybe get a more accurate number is by making the model
do "warm-up". By making the model do warm-up, we can get a better generalize performance from the model.

Without warm-up:

![without_warm_up](https://github.com/user-attachments/assets/52053299-fcaa-48c3-b143-3d9846b4660b)


After using warm-up:

![cpu_torch_compile](https://github.com/user-attachments/assets/897f0a4a-0cf3-44d0-a959-03f961f6d974)


### 1B
Using Cuda.

![cuda_torch_compile](https://github.com/user-attachments/assets/75c1c69a-5419-4a7b-9f75-d11a9fe4aa96)

As we can see, both original and optimized model improved tremendously in performance.

### 2A
Compared using time and CPU used:

![cpu_torch_compile](https://github.com/user-attachments/assets/d1d048ab-f705-4bfe-91ea-69801113e231)

Compared using torch.utils.benchmark and CPU used:

![benchmark_cpu](https://github.com/user-attachments/assets/f2dc9ac5-a231-433a-bc48-eead065df747)


### 2B

Compared using time and Cuda used:

![using_time_cuda](https://github.com/user-attachments/assets/25cff6a8-6e48-4b08-a40d-db376ad2fa4f)

Compared using torch.utils.benchmark and Cuda used:

![benchmark_cuda](https://github.com/user-attachments/assets/d45ea59d-9ade-4022-8cd7-e9fcd1ecf5db)

Based on the data given, involving cuda significantly boost their performance either using time or torch.utils.benchmark as metrics.
In general, fused kernel SDPA achieve better results than its naive version.

### 3A
### 3B
### 3C
### 3D







