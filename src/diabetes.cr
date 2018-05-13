require "csv"
require "shainet"

# use binary dummy columns for `Outcome` column
label = {
  "0" => [1.to_f64, 0.to_f64],
  "1" => [0.to_f64, 1.to_f64],
}

# data structures to hold the input and results
inputs = Array(Array(Float64)).new
outputs = Array(Array(Float64)).new

# read the file
raw = File.read("./data/diabetes.csv")
csv = CSV.new(raw, headers: true)

# we don't want these columns so we won't load them
headers = csv.headers.reject { |h| ["SkinThickness", "DiabetesPedigreeFunction", "Outcome"].includes?(h) }

# load the data structures
while (csv.next)
  row_arr = Array(Float64).new
  headers.each do |header|
    row_arr << csv.row[header].to_f64
  end
  inputs << row_arr
  outputs << label[csv.row["Outcome"]]
end

# normalize the data
training = SHAInet::TrainingData.new(inputs, outputs)
training.normalize_min_max

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
diabetes.train(training.data.shuffle, :sgdm, :mse, epoch = 1000, threshold = -1.0, log = 100)

# save to file
diabetes.save_to_file("./model/diabetes.nn")

tn = tp = fn = fp = 0

# determine accuracy
training.normalized_inputs.each_with_index do |test, idx|
  results = diabetes.run(test)
  if results[0] < 0.5
    if outputs[idx][0] == 0.0
      tn += 1
    else
      fn += 1
    end
  else
    if outputs[idx][0] == 0.0
      fp += 1
    else
      tp += 1
    end
  end
end

puts "Training size: #{outputs.size}"
puts "----------------------"
puts "TN: #{tn} | FP: #{fp}"
puts "----------------------"
puts "FN: #{fn} | TP: #{tp}"
puts "----------------------"
puts "Accuracy: #{(tn + tp) / outputs.size.to_f}"

# Pregnancies,Glucose,BloodPressure,Insulin,BMI,Age
results = diabetes.run(training.normalize_inputs([0, 100, 140, 0, 25, 50]))
puts "There is a #{(training.denormalize_outputs(results)[1] * 100).round} percent chance you will have diabetes"
