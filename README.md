# ML Pipeline

This repository contains a machine learning pipeline for any ML task, e.g., image classification, sentiment analysis, etc... The pipeline is designed to streamline the process of training, evaluating, and deploying machine learning models

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
