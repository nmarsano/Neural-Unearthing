### Project Update (3/25/2021)
We spent a good portion of time trying to get the Flickr image scraping code from one of our researched papers to work, but found ourselves running into more and more issues the further we got along. We decided to shift focus to using a pre-existing dataset. We have downloaded the image dataset that we will train our CNN with. This image dataset (500 GB) we downloaded focused on scene classification as opposed to geographical location classification. As such, we have pivoted our project to identify scenes, for example "a village,"  "a watering hole," "a garden", etc. We are now trying to build a CNN to use for our image classification problem.

We will create our neural network using Pytorch in the Pomona HPC servers and train it with the dataset of images we collected. The type of neural network we will use is a convolutional neural network. Our type of inputs will be .jpeg files. We will be performing a classification on those .jpeg files. 

Our intended output will be a calculated probability matrix of scenes that are the highest likelihood for the input. Our end result is a web-based application to classify the scene of an image. 

We intend it to perform better than chance. To be more specific, this paper (http://graphics.cs.cmu.edu/projects/im2gps)  states that for every 6 guesses for each query, the median error is within 500 kilometers to the correct location. We would like our model to be around there. 

Issues we've run into so far:
- We struggled to get the Flickr scraping code to work, but found a pre-existing dataset of images that we changed our focus onto.
- Many of our literature and sources of research have achieved the exact goal we are aiming towards, many of them done by PhD students and professional computer science researchers, who have much more experience compared to us. 
- Having downloaded our image dataset of 500GB,  we realized that rather than a training set of location classification images, they are actually scene classification images. As such, we have decided to pivot our project goal from location classification to scene classification.


### Updated Literature Review (3/18/2021)
There is a various collection of pre-existing studies that have attempted to tackle this very problem, each proposing their own algorithms or different versions of similar algorithms. From these we took inspiration and molded our ideas as we researched our task from our initial plan to what we have now.

http://graphics.cs.cmu.edu/projects/im2gps/im2gps.pdf

The IM2GPS article performed a similar project. They set out with the goal to estimate geolocation from a single image via training a neural network on 6 million images scraped from Flickr and labeled with geotags. Using a nearest neighbor algorithm to compare individual photos with each of their 6 million images, comparing features such as line features, tiny images, color histograms built from the images, and more, they displayed the geographic location of a photo as a probability distribution over the Earth’s surface. 
