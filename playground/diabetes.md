# Deep Learning using Crystal

This workbook demonstrates how to create a NN using SHAInet.  We will be using the [Pima Indians dataset](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)

## Load and Parse Data

We need to parse the data and populate a couple data structures:
```playground
require "csv"

# use binary dummy columns for `Outcome` column
label = {
  "0" => [1.to_f64, 0.to_f64],
  "1"  => [0.to_f64, 1.to_f64],
}

# data structures to hold the data
inputs = Array(Array(Float64)).new
outputs = Array(Array(Float64)).new

# read the file
raw = File.read("./data/diabetes.csv")
csv = CSV.new(raw, headers: true)

# reject columns we don't want
headers = csv.headers.reject {|h| ["SkinThickness", "DiabetesPedigreeFunction", "Outcome"].includes?(h)}

# load the data
while(csv.next)
  row_arr = Array(Float64).new
  headers.each do |header|
    row_arr << csv.row[header].to_f64
  end
  inputs << row_arr
  outputs << label[csv.row["Outcome"]]
end
```

## Create the Model

We will now create a model using SHAInet:
```playground
require "shainet"

diabetes : SHAInet::Network = SHAInet::Network.new
diabetes.add_layer(:input, 6, :memory)
diabetes.add_layer(:hidden, 8, :memory)
diabetes.add_layer(:output, 2, :memory)
diabetes.fully_connect
```

## Train and Save the Model

Now lets put the pieces together and train the model:
```playground
require "csv"
require "shainet"

# use binary dummy columns for `Outcome` column
label = {
  "0" => [1.to_f64, 0.to_f64],
  "1"  => [0.to_f64, 1.to_f64],
}

# data structures to hold the input and results
inputs = Array(Array(Float64)).new
outputs = Array(Array(Float64)).new

# read the file
raw = File.read("./data/diabetes.csv")
csv = CSV.new(raw, headers: true)

# we don't want these columns so we won't load them
headers = csv.headers.reject {|h| ["SkinThickness", "DiabetesPedigreeFunction", "Outcome"].includes?(h)}

# load the data
while(csv.next)
  row_arr = Array(Float64).new
  headers.each do |header|
    row_arr << csv.row[header].to_f64
  end
  inputs << row_arr
  outputs << label[csv.row["Outcome"]]
end

# normalize the data
normalized = SHAInet::TrainingData.new(inputs, outputs)
normalized.normalize_min_max

# create a network
diabetes : SHAInet::Network = SHAInet::Network.new
diabetes.add_layer(:input, 6, :memory)
diabetes.add_layer(:hidden, 8, :memory)
diabetes.add_layer(:output, 2, :memory)
diabetes.fully_connect

# params for sgdm
diabetes.learning_rate = 0.01
diabetes.momentum = 0.01


# train the network
diabetes.train_batch(normalized.data, :sgdm, :mse, epoch = 1000, threshold = -1.0, log = 100)

# save the model
diabetes.save_to_file("./model/diabetes.nn")
```

## Load and Run the Model

Now we can load and run against the trained model:
```playground
require "shainet"

diabetes : SHAInet::Network = SHAInet::Network.new
diabetes.load_from_file("./model/diabetes.nn")

# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

# Pregnancies,Glucose,BloodPressure,Insulin,BMI,Age
results = diabetes.run(training.normalize_inputs([1,85,66,0,26.6,31]))
puts "There is a #{(training.denormalize_outputs(results)[1] * 100).round} percent chance you will have diabetes"```
