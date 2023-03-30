Cart Recommendation Transformer
===============================

The Cart Recommendation Transformer is a Machine Learning model that predicts items that may be of interest to a user based on the existing items in the shopping cart. The model is trained on a set completion task by masking certain items in the cart and then predicting the missing item(s) using a Transformer architecture. It can be applied by any ecommerce business to make recommendation to the users who view their shopping carts. 

### Datasets
Example training/evaluation datasets are provided in the `input/` directory which includes 2 dataframes:
1. cart.parquet: A cart level pandas dataframe with 2 required columns
    - "cart_id" as the primary key
    - "test_row_indicator" with 3 unique values: "TRAIN", "VAL", and "TEST"
2. cart_item.parquet: A cart-item pair level pandas dataframe
    - index: cart_id (for lookup purpose)
    - 2 columns:
        - product_idx: The item identifier. Each unique item is represented by an integer ranging from [1: num_unique_items]. Unknown items are encoded with 0s.
        - category_idx: The category of the product. Each unique category is represented by an integer ranging from [1: num_unique_category]. Unknown categories are encoded with 0s.
In the example, the only item attribute is the `category`. The code can be easily modified to include more item attributes.

### User Guide
```
# clone the repo
git clone https://github.com/ShinSiangChoong/cart_recommendation_transformer.git

# change directory
cd cart_recommendation_transformer

# install the src module
pip install -e .

# training
python src/train.py --epochs 50 --cart_path './input/cart.parquet' --cart_item_path './input/cart_item.parquet'
# You can easily change other hyperparameters by checking out the `parse_args` function in src/train.py

# Evaluation
python src/eval.py --cart_path './input/cart.parquet' --cart_item_path './input/cart_item.parquet'