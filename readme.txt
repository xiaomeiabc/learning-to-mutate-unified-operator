***************************************************************************************************
The code for ''Learning Unified Mutation Operator for Differential Evolution by Natural Evolution Strategies.''

Haotian Zhang et al.  Dec 1st 2022             zht570795275@stu.xjtu.edu.cn

***************************************************************************************************

It contains three main algrithms:

1.  UCDE.m, test_UCDE.m, crossover_UDE.m and mutation_UDE.m are functions for Ada-UCDE.

2. UJADE.m, test_UJADE.m are functions for Ada-UJADE.

3. ULSHADE.m, test_ULSHADE.m are functions for Ada-ULSHADE.

***************************************************************************************************

To test the three algorithm, 

First, mex the cec 2017 test function (provided in CECtest_func), it can also be found at 
https://github.com/P-N-Suganthan/CEC2017-BoundContrained.git. 
The procedure please find in the codes_cec2017.rar.

Second run the test functions test_UCDE.m or test_UJADE.m or test_ULSHADE.m. 

Note that the methods need trained neural networks, we provide the trained network in UCDE_net, UJADE_net and ULSHADE_net.

The networks for Ada-UCDE and Ada-UJADE are for 30D test functions. And the networks for Ada-ULSHADE are for 50D test functions. 

The hyper-parameters are provided in the codes.

