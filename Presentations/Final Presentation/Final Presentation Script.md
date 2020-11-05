My approach is similar to Sakshi, on answer uncanny valley is to classify stories first.

Obviously, traditional programming techniques where we only writing rules for the programs wouldn't let us go very far because stories have tons of variations, especially we are using stories online. Also, training embeddings ourselves can be very tedious considering we have 15,000 stories, 30-50 sentences at least for one subreddit alone. So we are using sbert pretrained model. Their models are evaluated extensively and achieve state-of-the-art performance on various tasks. Further, the code is tuned to provide the highest possible speed.

For the data processing step, we are using sbert pretrained model and combining its embeddings with numerical values in the dataset, such as scores or comment scores. Take them into account. Finally, we have somewhat high-dimensional embeddings for each story.

However, one question may raise like are these vectors going separate? If these stories are mixed in a chaotic way that are not going to separate at all, doing neural networks on it is completely useless. So, one way to visualize these high dimensional data is using t-SNE. After some parameters tuning, our data are separated quite easily with very few iterations.

next slide:

Move on to our keras model. I tried some hyperparameters and fit our model. Already, it gave us above at least 85% accuracy for training and validation data. Then, I did some hyperparameter tunings using a randomized search algorithm. It pushed our accuracy even further, up to 96%. We can see these graphs from tensorboard, we have big accuracies increase and losses decrease.

next slide:

Finally, we tried some real-world sentences. we can see that the prediction is right on point. The first text is creepy and it gives 1 being in the class of creepy while the latter gives us a small number which leans toward noncreepy.

---

Ok here are our conclusions. We found out that certain words are strongly associated with creepy/non-creepy text. Also, we can accurately classify text as creepy using deep neural networks. Finally, we found out that humans might interact with known bots differently than other humans.