# Deep Learning Example using SHAInet 

This workbook demonstrates how to create a Deep Learning network using [SHAInet](https://github.com/NeuraLegion/shainet).  We will be using the [Pima Indians dataset](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes) to predict diabetes.

## Installation

This requires crystal 0.23.1

## Usage

This project uses crystal's playground.  You can load and run the playground workbook using:
```bash
shards install
crystal play
open http://localhost:8080
```
Then select the Workbook -> Diabetes from the menu.

You can also compile and run the application:
```bash
crystal run src/diabetes.cr
```

## Contributing

1. Fork it ( https://github.com/drujensen/diabetes/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [drujensen](https://github.com/drujensen) Dru Jensen - creator, maintainer
