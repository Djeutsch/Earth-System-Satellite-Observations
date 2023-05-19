##### Load the required pakackes and modules
import lightgbm as lgbm
from sklearn.model_selection import train_test_split

##### Create an Light Gradient Boosting Machine (LGBM) classifier
    

def lgbm_clissifier(train_x, train_y, mb_round=100, verbose_eval=10):
    """
    Creates and trains an LGBM classifier on the training input data
    
    parameters:
        train_x: feature data in DataFrame or array-like of shape
            (n_samples, n_features)
        train_y: target data in DataFrame or array-like of shape
            (n_samples,)
        mb_round: number of boost round or epochs
        verbose_eval: bool (default True) to inform during training
            
    Returns:
        the trained lgbm classifier that can be used to make predictions
    """
    ### Set the training dataset
    trn_x, val_x, trn_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2)
    trn_data = lgbm.Dataset(trn_x, label=trn_y)
    val_data = lgbm.Dataset(val_x, label=val_y)

    ### Define its hyper-parameters
    params = {}
    params["learning_rate"] = 0.025
    params["boosting_type"] = "gbdt"  # Gradient Boosting Decision Tree
    params["objective"] = "multiclass"  # Multi-class target feature (6)
    params["metric"] = "multi_logloss"  # metric for multi-class
    # No. of classes or land cover types: 5 + 1 unlabel
    params["num_class"] = 6
    params["drop_rate"] = 0.9
    params["max_bin"] = 256
    params["max_depth"] = 16
    params["num_leaves"] = 32
    params["sub_feature"] = 0.50      # feature_fraction
    params["bagging_fraction"] = 0.85  # sub_row
    params["bagging_freq"] = 40
    params["min_data"] = 512         # min_data_in_leaf
    params["min_hessian"] = 0.05     # min_sum_hessian_in_leaf
    params["force_col_wise"] = True  # In case the memory is not enough.
    params["verbose"] = 0

    #### Train or fit the classifier
    print(f"\n\nTraining of the LGBM classifier ...\n")
    lgbm_clf = lgbm.train(params=params, train_set=trn_data,
                          valid_sets=val_data, early_stopping_rounds=10,
                          num_boost_round=mb_round, verbose_eval=verbose_eval)

    return lgbm_clf
