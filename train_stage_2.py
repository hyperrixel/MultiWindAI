from csv import reader
from datetime import datetime
import numpy as np
import pandas as pd
from random import shuffle
from os.path import isfile, join
import re
import torch

""" X """

DATA_ROOT = 'data'                  # Data root directory - can be different
DATASET_DIR = 'Dateset#{}'          # Dataset subfolder mask

FORECAST_AHEAD = 8                  # Number of hours of viewable forecast
FORECAST_AHEAD_DIFFERENCE = 0       # T+AHEAD_DIFFERENCE th forecast will be added first
PREDICTION_FREQUENCY = 300          # Time delta in seconds of making prediction
PREDICTION_INTERVAL_COUNT = 73      # Number of 5 minute predictions to make
PREDICTION_GAP = 5                  # Number of 5 minute units to leave out.
PREDICTION_TIME_MAX_DELTA = 11      # Maximum delay to a whole hour prediction
                                    # in 5 minute time units.

NORMALIZE_WIND_SPEED_TO = 20        # Maximum wind speed in m/s to normalize to

""" X """

SELECTED_MODEL_ID = 1               # The ID of the model which is selected
USE_ACTUAL_MEASURE = True           # Whether or not to use the actual
                                    # (non-historical) measure data
NUM_EPOCHS = 100                    # Count of epochs to train
BATCH_SIZE = 128                    # Count of batches in a backpropagation

""" X """

TRAIN_ID = 1                        # ID of the dataset 1 (train dataset)
TUNE_ID = 2                         # ID of the dataset 2 (tuning dataset)
EVALUATION_ID = 3                   # ID of the dataset 3 (evaluation dataset)
DATETIME_FORMAT = '%Y-%m-%d %H:%M'  # Timestamp conversion string

FORECAST_INTERVAL = 3600            # Interval of forecast in seconds

""" X """

class Model_1(torch.nn.Module):
    """
    This class represents a deep learning model.
    """



    def __init__(self):
        """
        Initializes the object
        ======================

        Notes
        -----
            This model ...
        """

        super(self.__class__, self).__init__()
        self.fc1 = torch.nn.Linear(18, 256)
        self.fc2 = torch.nn.Linear(256, 1024)
        self.fc3 = torch.nn.Linear(1024, 256)
        self.fc4 = torch.nn.Linear(256, 1)
        self.drop = torch.nn.Dropout(p=0.1, inplace=False)




    def forward(self, x):
        """
        Performs a forward pass
        =======================

        Parameters
        ----------
        x : torch.tensor
            Input for prediction.

        Returns
        -------
        torch.Tensor
            The prediction.
        """

        x = self.drop(torch.nn.functional.relu(self.fc1(x), True))
        x = self.drop(torch.nn.functional.relu(self.fc2(x), True))
        x = self.drop(torch.nn.functional.relu(self.fc3(x), True))
        return self.fc4(x)

""" X """

class Model_2(torch.nn.Module):
    """
    This class represents a deep learning model.
    """



    def __init__(self):
        """
        Initializes the object
        ======================

        Notes
        -----
            This model ...
        """

        super(self.__class__, self).__init__()
        self.cv1 = torch.nn.Conv1d(1, 32, 1)
        self.cv2 = torch.nn.Conv1d(32, 64, 3)
        self.cv3 = torch.nn.Conv1d(64, 64, 1)
        self.fc1 = torch.nn.Linear(896, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.drop = torch.nn.Dropout(p=0.1, inplace=False)




    def forward(self, x):
        """
        Performs a forward pass
        =======================

        Parameters
        ----------
        x : torch.tensor
            Input for prediction.

        Returns
        -------
        torch.Tensor
            The prediction.
        """

        x = x.unsqueeze(1)
        x = torch.nn.functional.relu(self.cv1(x), True)
        x = torch.nn.functional.relu(self.cv2(x), True)
        x = torch.nn.functional.relu(self.cv3(x), True)
        x = torch.flatten(x, 1)
        x = self.drop(torch.nn.functional.relu(self.fc1(x), True))
        return self.fc2(x)

""" X """

class Forecast(object):
    """
    """



    def __init__(self, wind_speed, wind_dic):
        """
        """

        self.__wind_speed = float(wind_speed)
        self.__wind_dic = float(wind_dic)



    @property
    def wind_dic(self):
        """
        """

        return self.__wind_dic



    @property
    def wind_speed(self):
        """
        """

        return self.__wind_speed



    def __repr__(self):
        """
        """

        return 'Forecast({}, {})'.format(self.wind_speed, self.wind_dic)



    def __str__(self):
        """
        """

        return 'Forecast: windSpeed: {} (m/s); windDic: {} (degree)'.format(self.wind_speed,
                                                                            self.wind_dic)


class Measurement(object):
    """
    """



    def __init__(self, available_mw, wind_speed, wind_power):
        """
        """

        self.__available_mw = float(available_mw)
        self.__wind_speed = float(wind_speed)
        self.__wind_power = float(wind_power)
        # However we cleaned the dataset, but in real world you can never be too careful.
        if self.available_mw != 0:
            self.__normalized_wind_power = self.wind_power / self.available_mw
            if self.normalized_wind_power > 1:
                self.__normalized_wind_power = 1.0
        else:
            self.__normalized_wind_power = 0.0



    @property
    def available_mw(self):
        """
        """

        return self.__available_mw



    @property
    def normalized_wind_power(self):
        """
        """

        return self.__normalized_wind_power



    @property
    def wind_power(self):
        """
        """

        return self.__wind_power



    @property
    def wind_speed(self):
        """
        """

        return self.__wind_speed



    def __repr__(self):
        """
        """

        return 'Measurement({}, {}, {})'.format(self.available_mw,
                                                self.wind_speed,
                                                self.wind_power)



    def __str__(self):
        """
        """

        return 'Measurement: availableMW: {}; windSpeed: {}; windPower{} ({})'.format(self.available_mw,
                                                                                      self.wind_speed,
                                                                                      self.wind_power,
                                                                                      self.normalized_wind_power)



class DataPoint(object):
    """
    """



    def __init__(self, forecasts, measurements):
        """
        """

        self.__forecasts = forecasts
        self.__forecast_count = len(self.__forecasts)
        self.__measurements = measurements
        self.__raw_measurement_count = len(self.__measurements)
        self.__measurement_count = 0
        for measurement in self.__measurements:
            if measurement is not None:
                self.__measurement_count += 1



    @property
    def forecasts(self):
        """
        """

        return self.__forecasts[:]



    @property
    def forecast_count(self):
        """
        """

        return self.__forecast_count



    @property
    def get_forecast(self, id):
        """
        """

        if id < self.forecast_count:
            return self.__forecasts[id]
        else:
            raise IndexError('Forecast id is out of range.')



    def get_measurement(self, id):
        """
        """

        if id < self.raw_measurement_count:
            return self.__measurements[id]
        else:
            raise IndexError('Measurement id is out of range.')



    @property
    def measurements(self):
        """
        """

        return self.__measurements[:]



    @property
    def measurement_count(self):
        """
        """

        return self.__measurement_count



    @property
    def raw_measurement_count(self):
        """
        """

        return self.__raw_measurement_count



    def __repr__(self):
        """
        """

        return 'DataPoint({}, {})'.format(self.__forecasts,
                                              self.__measurements)



    def __str__(self):
        """
        """

        return 'DataPoint objact with {} forecasts and {} measurements ({} raw)'.format(self.forecast_count,
                                                                                        self.measurement_count,
                                                                                        self.raw_measurement_count)



class DataRow(object):
    """
    """



    def __init__(self, x, y=None):
        """
        """

        self.__x = x
        self.__y = y



    @property
    def x(self):
        """
        """

        return self.__x



    @property
    def y(self):
        """
        """

        return self.__y

""" X """

def get_batches(data, batch_size, drop_fragment=False):
    """
    Gets batches from the data
    ==========================

    Parameters
    ----------
    data : list of DataRow objects
        The data to slice to batches.
    batch_size : int
        Count of data elements in a batch.
    drop_fragment : bool, optional (True if omitted)
        Whether to drop non-whole batches or not.

    Yields
    ------
    tuple of list
        Batch data. The first element is the input and the second element is
        the target. Batch size is the first dimension of each element.
    """

    len_data = len(data)
    pos = 0
    while pos + batch_size < len_data:
        records = data[pos:pos + batch_size]
        x_values = [record.x for record in records]
        y_values = [record.y for record in records]
        yield x_values, y_values
        pos += batch_size
    if not drop_fragment:
        records = data[pos:]
        x_values = [record.x for record in records]
        y_values = [record.y for record in records]
        yield x_values, y_values



def get_raw_data(stage_id):
    """
    Gets raw data from file
    =======================
    Parameters
    ----------
    stage_id
        ID of the dataset to load.

    Returns
    -------
    tuple
        Raw data: forecast, measurements

    Notes:
        Both forecast and measurements are dicts. The key is the
        timestamp which belongs to the data.
    """

    global DATA_ROOT
    global DATASET_DIR
    global DATETIME_FORMAT

    filename = join(DATA_ROOT, DATASET_DIR.format(stage_id),
                    'Dataset{}_forecast.csv'.format(stage_id))
    forecasts = {}
    if isfile(filename):
        file_df = pd.read_csv(filename, delimiter =',')
        raw_data = file_df.values.tolist()
        for row in raw_data:
            if re.search(r'\s\d:', row[0]) is not None:
                key = row[0].replace(' ', ' 0')
            else:
                key = row[0]
            key = int(datetime.strptime(key, DATETIME_FORMAT).timestamp())
            forecasts[key] = Forecast(row[1], row[2])
    else:
        print('Failed to load forecast data.')

    filename = join(DATA_ROOT, DATASET_DIR.format(stage_id),
                    'Dataset{}_measurement.csv'.format(stage_id))
    measurements = {}
    if isfile(filename):
        file_df = pd.read_csv(filename, delimiter =',')
        for column in file_df.columns:
            if column != 'timeStamp':
                file_df[column] = pd.to_numeric(file_df[column], errors='coerce')
        file_df = file_df.dropna()
        file_df = file_df[file_df['availableMW'] != 0]
        file_df['ratio'] = file_df['windPower'] / file_df['availableMW']
        indexnames = file_df[(file_df['ratio'] > 1.0) & (file_df['windSpeed'] < 12)].index
        file_df.drop(indexnames, inplace = True)
        file_df.drop(columns='ratio')
        raw_data = file_df.values.tolist()
        for row in raw_data:
            if row[1] != '\\N' and row[2] != '\\N':
                if re.search(r'\s\d:', row[0]) is not None:
                    key = row[0].replace(' ', ' 0')
                else:
                    key = row[0]
                key = int(datetime.strptime(key, DATETIME_FORMAT).timestamp())
                measurements[key] = Measurement(row[1], row[2], row[3])
    else:
        print('Failed to load measure data.')
    return forecasts, measurements



def make_dataset(forecasts, measurements):
    """
    Creates dataset
    ===============

    Parameters
    ----------
    forecasts
        A
    measurements
        A

    Returns
    -------
    list
        Dataset combined from data sources to train with it.
    """

    # Globals are here due to more compatibility and readabilty of the code.
    global FORECAST_AHEAD
    global FORECAST_AHEAD_DIFFERENCE
    global FORECAST_INTERVAL
    global PREDICTION_FREQUENCY
    global PREDICTION_GAP
    global PREDICTION_INTERVAL_COUNT
    global PREDICTION_TIME_MAX_DELTA

    print('[.] make_dataset() info:')
    x_base = {}
    # Because of the dataset sorted is not required but useful for future use.
    for timestamp in sorted(forecasts.keys()):
        forecast_container = []
        for i in range(FORECAST_AHEAD_DIFFERENCE,
                       FORECAST_AHEAD + FORECAST_AHEAD_DIFFERENCE):
            lookup_key = timestamp + (FORECAST_INTERVAL * i)
            if lookup_key in forecasts.keys():
                forecast_container.append(forecasts[lookup_key])
        if len(forecast_container) == FORECAST_AHEAD:
            x_base[timestamp] = forecast_container
    print('--- From {} forecast(s) {} forecast window made.'
          .format(len(forecasts), len(x_base)))
    datapoints = {}
    perfect_window_counter = 0
    perfect_window_length = PREDICTION_INTERVAL_COUNT + PREDICTION_TIME_MAX_DELTA
    window_length = PREDICTION_GAP + PREDICTION_INTERVAL_COUNT + PREDICTION_TIME_MAX_DELTA
    for timestamp in x_base.keys():
        measurement_container = []
        found_keys = 0
        # +1 is required since our first prediction is T+1 prediction.
        for i in range(PREDICTION_GAP + 1, window_length + 1):
            lookup_key = timestamp + (PREDICTION_FREQUENCY * i)
            if lookup_key in measurements.keys():
                measurement_container.append(measurements[lookup_key])
                found_keys += 1
            else:
                measurement_container.append(None)
        if found_keys == perfect_window_length:
            perfect_window_counter += 1
        datapoints[timestamp] = DataPoint(x_base[timestamp], measurement_container)
    print('--- From {} forecast window(s) {} is perfect.'
          .format(len(x_base), perfect_window_counter))
    print('--- {} datapoint(s) made.'.format(len(datapoints)))
    return datapoints



def make_records(datapoints, for_train=True):
    """
    Makes dataset records
    =====================

    Parameters
    ----------
    datapoints : list of DataPoint objects
        The datapoints to use for making data records.
    for_train : bool, optional (True if omitted)
        Whether to creat target values or not.

    Returns
    -------
    list
        Data to use with the model.

    Notes
    -----
        The structure of the data node:
        Content of x:
        ID | Content
        ---+--------
        0  | wind speed forecast at T + FORECAST_AHEAD_DIFFERENCE + 0
           | Normalized from 0-NORMALIZE_WIND_SPEED_TO to 0-1
        1  | wind direction forecast at T + FORECAST_AHEAD_DIFFERENCE + 0
           | Normalized from 0-360 to 0-1
        ... up to FORECAST_AHEAD times
        +1 | Distance of prediction time from whole hour in 5 minute units
           | Normalized from 0-11 to 0-1
        +2 | Distance of target value to predict calculated from prediction time
           | Normalized from 0-PREDICTION_INTERVAL_COUNT to 0-1
        Content of y:
        value of normalized_wind_power at the given target time
    """

    global NORMALIZE_WIND_SPEED_TO
    global PREDICTION_INTERVAL_COUNT
    global PREDICTION_TIME_MAX_DELTA

    print('[.] make_records() info:')
    records = []
    for datapoint in datapoints.values():
        x_core = []
        for forecast in datapoint.forecasts:
            speed = forecast.wind_speed / NORMALIZE_WIND_SPEED_TO
            x_core.append(speed if speed <= 1 else 1.0)
            x_core.append(forecast.wind_dic / 360)
        for i in range(PREDICTION_TIME_MAX_DELTA):
            node_p1 = i / PREDICTION_TIME_MAX_DELTA
            for j in range(PREDICTION_INTERVAL_COUNT):
                node_p2 = j / PREDICTION_INTERVAL_COUNT
                target = datapoint.get_measurement(i + j)
                if target is not None:
                    x_data = x_core[:]
                    x_data.append(node_p1)
                    x_data.append(node_p2)
                    records.append(DataRow(x_data,
                                           y=target.normalized_wind_power))
    print('--- {} records made.'.format(len(records)))
    return records


""" X """

forecasts_train, measurements_train = get_raw_data(EVALUATION_ID)
datapoints_train = make_dataset(forecasts_train, measurements_train)
del forecasts_train
del measurements_train
data_train = make_records(datapoints_train)
del datapoints_train
exit()
""" X """

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device {}.'.format(device))
model = Model_1()
model.load_state_dict(torch.load('model_1_s2_full_013e_0.0400l.pth'))
model.to(device)
for x, y in get_batches(data_train, BATCH_SIZE):
    len_x = len(x)
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).unsqueeze(-1).to(device)
    optimizer.zero_grad()
    yhat = model(x)
    loss = criterion(yhat, y)
    loss.backward()
    optimizer.step()
    batch_loss = loss.item()
    epoch_loss += batch_loss * len_x
    batch_count += 1
    if batch_count % 1000 == 1:
        print('\r{}/{} epoch {} batch: batch loss {:0.6f}'
              .format(epoch, NUM_EPOCHS, batch_count, batch_loss), end='')
