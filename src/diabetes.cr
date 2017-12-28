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

# load the data structures
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
diabetes.train_batch(normalized.data, :adam, :mse, :sigmoid, 50000, 0.01)

# try it out and verify it works
diabetes.run(normalized.normalized_inputs.first)
diabetes.run(normalized.normalized_inputs.last)
