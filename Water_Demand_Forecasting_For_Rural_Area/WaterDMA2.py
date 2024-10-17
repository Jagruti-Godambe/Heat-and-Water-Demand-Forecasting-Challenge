import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Initialize the preprocessing class with paths to the data files.
class Water_data_preprocessing:
    def __init__(self, data_path, meter_data_path, weather_data_path):

      self.train_data = Water_data_preprocessing.load_Water_data(data_path)
      self.meter_data = Water_data_preprocessing.load_meter_data(meter_data_path)
      self.Weather_data = Water_data_preprocessing.load_weather_data(weather_data_path)

      self.scaler = MinMaxScaler()
      self.columns_to_scale = ['dew_point','feels_like','temp_min','temp_max','wind_speed',
                               'wind_deg','clouds_all','temp','pressure','humidity']

    @staticmethod
    # Load and preprocess water consumption data.
    def load_Water_data(data_path):
      data =  pd.read_csv(data_path)
      data.columns = ['timestamp', 'comsumption']
      data['timestamp'] = pd.to_datetime(data['timestamp'])

      return data

    @staticmethod

   # Load and preprocess weather data.
    def load_weather_data(data_path):
      Weather_data = pd.read_csv(data_path)
      Weather_data['dt_iso'] = Weather_data['dt_iso'].astype(str)
      Weather_data['dt_iso'] = Weather_data['dt_iso'].str.replace(' UTC', '', regex=False)
      Weather_data['dt_iso'] = pd.to_datetime(Weather_data['dt_iso'])
      Weather_data['timestamp'] = Weather_data['dt_iso'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
      Weather_data = Weather_data.drop(columns=['dt_iso', 'snow_1h','snow_3h', 'grnd_level',
                                                'sea_level','visibility','wind_gust','rain_1h','rain_3h'])
      Weather_data['timestamp'] = pd.to_datetime(Weather_data['timestamp'])

      return Weather_data

    @staticmethod
    # Load and preprocess meter data.
    def load_meter_data(data_path):
      data =  pd.read_csv(data_path)
      data.columns = ['timestamp', 'meters']
      data['timestamp'] = pd.to_datetime(data['timestamp'])
      return data

    def basic_data_preprocessing(self, data, if_train_data=True):
      # Merge water data, meter data, and weather data
      Water_data_with_meter_data = pd.merge(data, self.meter_data, on='timestamp', how='inner')
      Water_data_with_meter_data = pd.merge(Water_data_with_meter_data, self.Weather_data, on='timestamp', how='inner')
      
      # Interpolate consumption values if this is training data
      if if_train_data:
        Water_data_with_meter_data['comsumption']= Water_data_with_meter_data['comsumption'].interpolate(method='linear')
      
      # Calculate consumption per meter after interpolation
      Water_data_with_meter_data['Per_meter_comsumption_with_inter'] = Water_data_with_meter_data['comsumption'] / Water_data_with_meter_data['meters']

      #Drop un necessary columns
      Water_data_with_meter_data = Water_data_with_meter_data.drop(['weather_id','timezone',
                                                                  'city_name','lat','lon','dt'],axis=1)

      return Water_data_with_meter_data

    # Fit the scaler to the training data and apply transformations.
    def fit(self):

      Water_data_with_meter_data = self.basic_data_preprocessing(self.train_data)
      Water_data_with_meter_data[self.columns_to_scale] = self.scaler.fit_transform(Water_data_with_meter_data[self.columns_to_scale])
      Water_data_with_meter_data = Water_data_with_meter_data.set_index('timestamp')

      dummies = pd.get_dummies(Water_data_with_meter_data['weather_icon'],
                              prefix='weather_icon', dtype='int')

      self.dummies_features = dummies.columns.tolist()

      Water_data_with_meter_data = pd.merge(Water_data_with_meter_data, dummies, left_index=True, right_index=True)
      Water_data_with_meter_data = Water_data_with_meter_data.drop(['weather_main','weather_description','weather_icon'],axis=1)

      return Water_data_with_meter_data
    
 # Transform new data using the fitted scaler and preprocessing steps.
    def transform(self, data_path):
      test_data = Water_data_preprocessing.load_Water_data(data_path)
      Water_data_with_meter_data = self.basic_data_preprocessing(test_data, if_train_data=False)
      Water_data_with_meter_data[self.columns_to_scale] = self.scaler.transform(Water_data_with_meter_data[self.columns_to_scale])
      Water_data_with_meter_data = Water_data_with_meter_data.set_index('timestamp')
      # Create one-hot encoded features for weather_icon
      dummies = pd.get_dummies(Water_data_with_meter_data['weather_icon'],
                              prefix='weather_icon', dtype='int')

      dummies = dummies[self.dummies_features]
      Water_data_with_meter_data = pd.merge(Water_data_with_meter_data, dummies, left_index=True, right_index=True)
      Water_data_with_meter_data = Water_data_with_meter_data.drop(['weather_main','weather_description','weather_icon'],axis=1)
      return Water_data_with_meter_data

    def get_scaler(self):
       return self.scaler
    
    def get_columns_to_scale(self):
       return self.columns_to_scale



# Function to create sequences and corresponding labels
def create_sequences(data, sequence_length_x, sequence_length_y, data_scaling_info):

    sequences = []
    labels = []
    scaling_prameters = []

   # Loop through the data to generate sequences and labels
    for i in tqdm(range(len(data) - (sequence_length_x + sequence_length_y))):
        sequence = data[i:i + sequence_length_x]#.drop(['comsumption','meters'], axis=1)

        label_idx = i + sequence_length_x
        label = data['Per_meter_comsumption_with_inter'][label_idx : label_idx + sequence_length_y]
        scaling_prameter = data_scaling_info[label_idx : label_idx + sequence_length_y]

        sequences.append(np.array(sequence))
        labels.append(np.array(label))
        scaling_prameters.append(np.array(scaling_prameter))

    return np.array(sequences), np.array(labels), np.array(scaling_prameters)


# Split sequences and labels into training and testing sets.
def train_test_split(sequences, labels, data_scaling_info, train_size = 0.8):

  train_size_len = int(sequences.shape[0] * train_size)

  return (sequences[:train_size_len,:,:], labels[:train_size_len,:], data_scaling_info[:train_size_len,:,:],
          sequences[train_size_len:,:,:], labels[train_size_len:,:], data_scaling_info[train_size_len:,:,:])


class WaterData(Dataset):
    def __init__(self, sequences, labels):
      self.sequences = sequences
      self.labels = labels

    def __len__(self):
      return self.sequences.shape[0]

    def __getitem__(self, idx):
      return torch.Tensor(self.sequences[idx]), torch.Tensor(self.labels[idx])



# LSTM model for time series prediction.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def get_device():
    if torch.cuda.is_available():
      device = "cuda"
    elif torch.backends.mps.is_available():
      device = "mps"
    else:
      device = "cpu"

    return device



# Calculate the Mean Absolute Error (MAE).
def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    abs_errors = np.abs(y_true - y_pred)
    mae = np.mean(abs_errors)

    return mae


#Calculate the Mean Absolute Percentage Error (MAPE).
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    epsilon = np.finfo(float).eps
    y_true = np.where(y_true == 0, epsilon, y_true)
    abs_percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    mape = np.mean(abs_percentage_errors)
    
    return mape





