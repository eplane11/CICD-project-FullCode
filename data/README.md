# Data Directory

This directory is intended for storing the used car dataset required for the pipeline.

## Instructions

- Place your `used_cars.csv` file in this directory.
- The CSV should contain the following columns:
  - Segment
  - Kilometers_Driven
  - Mileage
  - Engine
  - Power
  - Seats
  - Price

**Note:**  
Do not commit sensitive or proprietary data to the repository.  
For AzureML, register this file as a data asset as described in the main README.

Example registration code:
```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

data_asset = Data(
    path="data/used_cars.csv",
    type=AssetTypes.URI_FILE, 
    description="A dataset of used cars for price prediction",
    name="used-cars-data"
)
ml_client.data.create_or_update(data_asset)