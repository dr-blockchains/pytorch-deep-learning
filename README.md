# Evaluation and Comparison of Different Neural Networks

As you know, Neural Networks (NNs) without a hidden layer (HL) are mathematically equivalent to generalized linear models (GLMs), with the activation function (AF) as the link function. For example, minimizing binary cross-entropy loss for a NN with a sigmoid output is equivalent to maximizing the likelihood function for logistic regression; except, regression coefficients are often faster to estimate and easier to interpret.

Adding a HL with a linear (or no) AF doesn't increase the model's expressive power, as multiple linear transformations can be collapsed into one. However, in cases of multicollinearity, a linear HL with fewer neurons than the minimum of inputs and outputs may serve as a supervised dimensionality reduction (like PLS).

NNs gain their true power via HLs with nonlinear AFs. According to the Universal Approximation Theorem, for any continuous function, there exists a feedforward NN with one HL that can approximate the function on compact subsets of ℝⁿ to arbitrary precision, given enough nonlinear neurons in the HL.

For simple tasks, one HL with a few ReLU neurons suffices. More complex tasks may benefit from more neurons and more layers, but excessive capacity (parameter count) can lead to overfitting. The challenge is to balance model capacity with generalizability while considering the available data and computational resources.

In the diagrams, I analyzed models with 5 inputs and 3 outputs. The NN with one linear HL has 3 neurons. The NN with 2 HLs has 8 and 8 ReLU neurons. There are 30 NNs with 1 to 30 hidden neurons in one HL. The one with 16 neurons has 147 trainable parameters, which happens to be equal to the number of parameters in the 2HL NN with 16 hidden neurons, making it a good case for comparison.

The barchart shows that the accuracy of y₁ (on the test set) is almost the same for all the models. That is because the underlying pattern for y₁ is linear and can be quickly approximated with few parameters.

The linear models perform equally poorly at predicting y₂ because it has nonlinearities. But the NNs with one HL (ReLU) perform better as the number of neurons increases up to about 6 neurons, which apparently can capture all the underlying complexity. More neurons do not improve it much, nor do more layers.

The linear models perform even worse on y₃ because it is more complex. The NNs with more ReLU neurons generally perform better. Interestingly, the NN with 16 ReLU neurons outperforms the 2HL NN with the same number of neurons, while 2HL NN takes longer to train due to backpropagation through layers. In most cases, a single HL with enough neurons can capture relevant patterns more efficiently than a deeper NN; but some specialized architectures such as CNNs for spatial data and LSTMs/Transformers for sequential data, achieve high predictive power with a large capacity by leveraging structural efficiencies.

To run this code, you need PyTorch and some other libraries. If you would like to add anything to this code, please fork this repo and open a PR. 
