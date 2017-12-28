# Deep Learning using Crystal

This workbook demonstrates how to create a NN using SHAINet.  We will be using the [Pima Indians dataset](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)

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

We will now create a model using SHAINet:
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

# train the network
diabetes.train_batch(normalized.data, :adam, :mse, :sigmoid, 10000, 0.01)

# save the model
diabetes.save_weights("./diabetes.h5")
diabetes.save_model("./diabetes.json")
```

## Load and Run the Model

Now we can load and run against the trained model:
```playground
require "shainet"

diabetes : SHAInet::Network = SHAInet::Network.new
diabetes.load_model("./diabetes.json")
diabetes.load_weights("./diabetes.h5")

diabetes.run([0.058823529411764705, 0.46733668341708545, 0.5737704918032788, 0.0, 0.45305514157973176, 0.03333333333333333])
```


