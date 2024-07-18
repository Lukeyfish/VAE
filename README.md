# Variational Auto Encoder

This repository is an example of a Variational Auto Encoder and how it effects the representational power of the data at the cost of a latent representation

## Table of Contents

- [Installation/Setup](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r environment.yml
```

To download fashion-mnist data run:

```bash
make data
```

## Usage

To run the model:

```bash
Make train
```

## Configuration

Edit the `configs/main.yaml` with data load path, and desired hyperparameters
