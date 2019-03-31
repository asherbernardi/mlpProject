# Testing the Multi Layer Perceptron

My algorithm did not work. I'm not sure why, since I'm convinced that it
followed the specifications. The outputs of the mlp were always approximately
0.5 after training. Regardless, I tested it on XOR, the Breast Cancer
dataset and the Mushrooms dataset.

## XOR
I consistently got 50% accuracy on this, which I attribute to randomness.

## Breast Cancer
I would get about 60% accuracy on this dataset, which one could argue is better
than the average, but I think is also just coincidence.

## Mushrooms
I got a 90% accuracy on this dataset, which seems good, but KNN was able to get
99%, so I think it may be some fluke.
