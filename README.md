# 2020 Hack Illinois Caterpillar IoT Machine Learning Analytics Challenge

Our attempt at the challenge utilizing isolation forests and DeepAnT.

Team Members: Samraj Moorjani, Nikash Walia, Aishani Dutta

## Inspiration

Caterpillar has a huge array of machines that collect millions of data points from their sensors. It's almost impossible to manually check to make sure all sensor readings are accurate which is why we use anomaly detection to determine what is working and what isn't. We can then use what we learn from channel predictions to help correct problems in data collection if there is a faulty sensor.

## What it does

Our model utilizes an isolation forest to locate all anomalies in our preprocessed dataset. This information is then sent to DeepAnT, a state-of-the-art deep learning approach for unsupervised anomaly detection in time series. We then use that to predict future times and we can further identify anomalies through the predictions made by DeepAnT.

## How we built it

Everything was built within Python. Our preprocessing was done with pandas and scipy. DeepAnT was built through Keras.

## Challenges we ran into

Much of this project involved familiarizing ourselves with the data and converting it into a usable format for our isolation forest and DeepAnT.

## Accomplishments that we're proud of

Having a working model which has a solid accuracy when comparing predicted data to target data.
