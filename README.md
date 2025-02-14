# Advance Deep Learning Labs Report

## Lab 1
### Questions:

![Questions_1](https://github.com/user-attachments/assets/94784592-a155-4b27-9c9e-40dfbbe4fc1f)

#### 1.1.A

![Lab1_1_A](https://github.com/user-attachments/assets/fe91235b-e30f-45ad-9ce9-b310ee08130f)

Based on the plot, we can see the accuracy stagnate after 8 fixed point widths. The highest is achieved during the value 8 at 83.9%


#### 1.1.B

![Lab1_1_B](https://github.com/user-attachments/assets/45192965-9609-4714-aa7a-0aa5683cccba)

The plot shows that using QAT in general is better than PTQ. QAT achieved its highest value at around 8 while PTQ at around 16. After those peaks, both PTQ and QAT value stagnate.

##### 1.2.A

![Lab1_2_A](https://github.com/user-attachments/assets/fbf038fb-e2f2-4294-b45c-7fe29744941c)

The graph shows that its highest accuracy during 0.1 and 0.2 sparsity. From then it gradually goes lower to around 0.50 at 0.9 sparsity.

#### 1.2.B

![Lab1_2_B](https://github.com/user-attachments/assets/d27d1ad1-66bc-4da4-bb6b-3392ceab44da)

The graph shows with different prunning strategies, L1-norm is still the best. Random methods stagnate a little above 0.5 from 0.1 to 0.9 sparsity.


## Lab 2
### Questions:

![question_2](https://github.com/user-attachments/assets/db1a047c-b629-4461-a116-e87ae86b8b06)

#### Lab 2.1.A and 2.1.B

![Lab2_1_A_B](https://github.com/user-attachments/assets/23842b96-cdbf-48f5-a413-c311996a2241)

From the plot, it can be seen that GridSampler and TPESampler has been used to measure the model performance.\
Based on their performance, GridSampler achieve the better result nearing 0.87 in its accuracy.


#### Lab 2.2.A and 2.2.B

Apologise sir/ma'am. I am unable to give a plot from this because my program always run into error. The error is eiter "Tensor found not on CPU/GPU" or "TypeError: forward() got an unexpected keyword argument 'token_type_ids'"

## Lab 3
### Questions:

![Questions_3](https://github.com/user-attachments/assets/c9e3c4cc-76b3-4210-9a6b-fa13ac148bc7)

#### Lab 3.1.A

![code_implementation_Lab_3_1](https://github.com/user-attachments/assets/42234dea-7543-4691-83df-2bb663a394e9)

Add an additional hyperparameter for the Optuna sampler to choose.

#### Lab 3.1.B


![lab3_B](https://github.com/user-attachments/assets/e2b07ada-2da9-48d3-90cd-3306df1c068e)

The plot shows that the highest accuracy is achieved from trial 13 onwards with accuracy above 0.879.

#### Lab 3.2.A

![list of linear layer implemented](https://github.com/user-attachments/assets/4a53ebef-b5f4-4678-b122-cde4d000a298)

The list of linear layer that will be implemented.

#### Lab 3.2.B

![plot_3_2_b](https://github.com/user-attachments/assets/8d7b9bd2-6251-47e8-bdf7-4b010232202c)

The graph above shows the results of the implemented linear layer. It shows a bit chaotic due to a combinatorial search space.\
A way that may help to overcome combinatorial search space is to classify the search space based on their types (arithmetics with each other and so on) 

## Lab 4
### Questions:
![Questions_4](https://github.com/user-attachments/assets/e1402b90-ebeb-4ef5-8f93-83a4f8daad96)

#### 4.1.A
There are many factors that can impact the run-time of torch.compile. Some of the factors included could be the size of the model, 
the batch size, the amount of operations executed and the CPU's ability. One way to maybe get a more accurate number is by making the model
do "warm-up". By making the model do warm-up, we can get a better generalize performance from the model.

Without warm-up:

![without_warm_up](https://github.com/user-attachments/assets/52053299-fcaa-48c3-b143-3d9846b4660b)


After using warm-up:

![cpu_torch_compile](https://github.com/user-attachments/assets/897f0a4a-0cf3-44d0-a959-03f961f6d974)


#### 4.1.B
Using Cuda.

![cuda_torch_compile](https://github.com/user-attachments/assets/75c1c69a-5419-4a7b-9f75-d11a9fe4aa96)

As we can see, both original and optimized model improved tremendously in performance.

#### 4.2.A
Compared using time and CPU used:

![cpu_torch_compile](https://github.com/user-attachments/assets/d1d048ab-f705-4bfe-91ea-69801113e231)

Compared using torch.utils.benchmark and CPU used:

![benchmark_cpu](https://github.com/user-attachments/assets/f2dc9ac5-a231-433a-bc48-eead065df747)


#### 4.2.B

Compared using time and Cuda used:

![using_time_cuda](https://github.com/user-attachments/assets/25cff6a8-6e48-4b08-a40d-db376ad2fa4f)

Compared using torch.utils.benchmark and Cuda used:

![benchmark_cuda](https://github.com/user-attachments/assets/d45ea59d-9ade-4022-8cd7-e9fcd1ecf5db)

Based on the data given, involving cuda significantly boost their performance either using time or torch.utils.benchmark as metrics.\
In general, fused kernel SDPA achieve better results than its naive version.

#### 4.3.A

The benefit MXINT8 could provide:
* More efficient in memory storage. In custom hardware where memory bandwith and storage is limited, this gives the memory an option
to store values in smaller format.
* Efficient Multiplication. Integer operations, like MXINT8, are usually faster and less resource heavy than floating-point operations.
* Parallelism. Custom hardware can take advantage of the lower bit-width to process more data in parallel


#### 4.3.B

Both are used on the calcuation of y[i].

dont_need_abs = a boolean value that is determined by the value mantissa_abs and 0x40. It handles whether the current value of hX[i]
                needs to be handled in a specific way or use the usual computation way. 
                
bias = used to calculate an offset that is applied to the result of out if dont_need_abs is false. bias works by setting fraction part out to zero, while           keeping the sign and exp same.  

#### 4.3.C
cta_tiller is a utility that helps partition data for optimal performance when copying between memory spaces.
It partitions the data into smaller manageable tiles that can be efficiently handled by the threads in a thread block.
The general steps on how cta_tiller partitions data for copying are:

<ol start="1">
  <li>Defining the tile size</li>
  <li>Partitioning the data</li>
  <li>Copying data to shared memory</li>
  <li>Handling strided or Non-contiguous Memory</li>
  <li>Memory coalescing and Data alignment</li>
  <li>Thread partitioning for computation</li>
</ol>

layout_sX partition is usually handled by the thread block's execution model to ensure efficient use of hardware recources.\
layout_sX refers to a method of how threads in a thread block is partitioned based on the X dimension. X here refers to the thread index in the thread block.
In general, it is a way of managing the organization of threads within a thread block to make computations more efficient by minimizing contention and optimizing resource utilization.

#### 4.3.D

Based on my search on the internet, here are some of hypothetical reasons:

* Approximation of Memory calculations. The calculations is not very precise into the final detail. Therefore some calculations is limited by approximation.
* Memory Fragmentation. This can lead to inefficiencies in memory usage such as effective savings could be lower than expected.
* Other varaibles. There may exist another varaibles that has not been taken into account during the calculation.
* GPU Memory Management. How the memory management works may caused some calculations to be off due to some specific operations.





