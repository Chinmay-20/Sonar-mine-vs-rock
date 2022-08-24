# Sonar-mine-vs-rock
Sonar rock vs mine prediction with sonar data in python.
There is a submarine in ocean. Enemy has planted mines in ocean. Submarine needs to find whether the object beneath is rock or mine. Submarine sends a sound signal called as sonar and receives it back. And we have to predict on basis of data of this signal. 

Workflow
1.We need to collect sonar data.
2.Data preprocessing
3.Train test split
4.Feed to logistic regression
5.Trained logistic regression model
6.Prediction


The data is collected in a lab in which metal cylinders are used.
We use logistic regression because it works well for binary classification problems. This problem is binary classification as we have to predict if the object is rock or mine.

R represents rock and m represents mine

There is a quite difference between mean values of mine and rock
We separate data using stratify . stratify separates the data based on number of output values.
