# Inspiration
## Background
We are deep learning developers and data scientists. We think, environmental protection is a really important topic. Choosing the challenge was difficult, since all of them were interesting and contained a lot of possibilities. Wind energy will be one of the most important energy sources in the future. However, wind is a renewable energy source, establishing and maintaining a wind farm is consuming a lot of other resources, such as money, time, human resource and capacity of the energy network. A small step for optimizing the wind energy production multiplies in the ecosystem that we called Earth. Every saved dolar can be spent for an other purpose. Making a good wind energy prediction can help to control the production of fossil or maybe nuclear power plants. A prediction time window from half hour to six and a half hours provides the possibility of an effective coordination of energy production.

## Vision
We want to build a wind energy prediction algorithm, based on wind speed and wind direction forecast, for wind farms to help to create a more sustainable power production system where the ratio of renewable energy sources will increase.

# What it does
## Intended Use Statement
MultiWInd AI creates wind energy predictions on a time window from half hour to six and a half hours. Predictions are made to 1 MW available energy production capacity. It is possible to run predictions not only hourly, but at any time.

## Indications for Use
Indications for use is making wind energy predictions on a time window from half hour to six and a half hours to 1 MW available energy production capacity. The software uses **8 sequential forecasts values**. Those mean the next 7 hours from the time of prediction. The first value refers to when the forecasted time equals the actual time (T+0). The next seven sequential forecast values mean the next following hours, such as T+1, T+2, … T+7. The values must be positive numbers where degree must be between 0 and 360. Any other normalization process is made by the algorithm itself.

## Limitations
The software is recommended to make wind energy predictions. Since the predictions are made to 1 MW available energy production capacity based on only forecasts of wind speed and wind directions, the software will fail if the wind farm is shutted down. The algorithm has no knowledge about the actual productivity of wind farms. This method helps to create a very flexible solution that can operate in several farms. 

The model is written in PyTorch, so the running machine have to meet the system requirements of PyTorch framework. 

# How we built it
We made Exploratory Data Analysis where we figured out that the dataset has values which cannot be converted into numbers, so they should be filtered out. During the EDA we discovered other interesting things. You can read after it in the GitHub page of the project.

## Preprocessing Steps
The pre-processor function of algorithm makes the following preprocessing steps:
normalize wind speed values between 0 and 1 where 1 means 20 m/s windspeed.
normalize wind direction between 0 and 1 where 1 means 360 degree

# Challenges we ran into
The given datasets contain invalid values such as non-digital values or records that slow the learning process or decrease the convergence. 

The data of the date was a small and interesting challenge on its own too. The hours were coded without leading zeros. Some programming languages and operating systems don't support this form of coding. Since we used Python on Windows we had that issue.

# What we learned
We learned about how wind farms operate. 

# What's next for MultiWind AI
We would like to continue this project to make a real life product from our idea. We have all the IT and deep learning skills to build up more complex algorithms.  To make this project real we need experts who can improve the effectiveness of our model with their own skills such as engineers, researchers and maintainers of wind farms. We are open to realize this project on its own or as a part of any other service as well. As a background we can imagine classical investment, venture capital investment or startup incubation.

# How we built the training records and why we did it
There is a data file of forecasts for each hour and there is a data file for measurements in each 5 minutes. The task is to create a prediction for each 5 minutes from 30 minutes to 6 hours 30 minutes after the creation of the prediction. The use of historical data is not allowed.

One of our first assumptions was that the T+0 measurement data is actually not historical. But we had to drop this assumption since in this case we couldn’t give predictions to dataset 3, where no measurements were given.

This means we have to make predictions based on forecasts only.

Our next question was to figure out how to imagine a real-life use-case. We decided to build a model which can predict in each 5 minutes based on 7 hour of forecasts only. The forecasts are given from T+0 to T+7 hours.

So we created forecast windows which contain 8 entries which are exactly one after one.

There are 7388 forecast entries in dataset 1 and 7311 windows can be made from those entries.

Now we have all the raw input data. Let’s add some calculated ones.

To make a perfect prediction system, maybe the best solution would be to create separated predictors for all use-cases. This means we have 12 opportunities in an hour to predict, and in each opportunity we have to predict for 73 target times. In case of the use of separated models it would mean 876 models (12 x 73).

We decided to not to do 876 train loops since we have a strong wish to finish this hackathon in-time.

So we decided to add two more data aside of the 14 raw onas, namely the relative position from the last whole hour and the absolute distance of the target prediction.

Finally our input data row consists of 16 elements.

There’s also another concept if we leave the in-hour relative distance.

If we look to the output there are also more approaches. First of all we have to decide how we produce the future predictions.

There are two main approaches; to produce them one-by-one or at the same time. Both concepts have their advantages and disadvantages as well. 
