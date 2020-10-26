# Slide 1

So this week, I created keras model for the pickle datasets I sent to you guys.

# Slide 2

So the data look like this. In particular, I am interested in the data from this column.

# Slide 3

After concatenating creepy and non creepy vectors together, it has around 38,000 instances and each instance has a vector of shape (769,). We can see that after concatenating, the labels of the first half are all ones and of the second half are all zeros, whereas 1 means creepy and 0 means non-creepy.

Then I shuffled the instances and labels correspondingly using the shuffle function from sklearn.

And then, I used the MinMaxScaler funciton from sklearn to map all values in the vectors to small values from 0 to 1.

Finally for the preprocessing step, I used 70% of data for training, 15% for validations, and the rest for testing.

# Slide 4

I used model that looks like this. I used 7 hidden layers. Since we are doing binary classification, we would use the 'sigmoid' activation function and we would use the 'binary_crossentropy' loss.

I also changed the optimizer to Adam instead of commonly used stochastic gradient descent. And finally, 30 epochs.

We can see that the accuracies are pretty high across training and validations

# Slide 5

However, after we plot the accuracies and losses into one graph. We can see that accuracy and loss of training are fine, but of validations, the accuracy and loss are fluctuating constantly.

# Slide 6

Here are other graphs from tensor board. The light blue and yellow lines were the actual accuracies and losses. If we smooth the lines we can see that validations does follow some patterns of training's accuracies and losses.

# Slide 7

Finally, evaluating on our test data, the result are pretty good, too.

# Slide 8

These result are good, but how to find better hyper parameters? I tried hyper parameter tuning using randomized search cv from sklearn. What it does is randmized search on hyper parameters in a loop and find the best hyper parameters after several trails and iterations.

It took a very long time to run even with a GPU, so I finished it after around 5-10 minutes. It gives me learning rate of 0.000959, 2 hidden layers and each layer has 303 neurons.

So I re-run the model as it said, it gives me less accuracy, but less fluctuations too. That is it. I think I will get back to hyperparameter tuning next week for more details.

# Slide 9

After all that, I randomly selected two stories from reddit to see how this model works. Unfortunately, it gives me completely different predictions.