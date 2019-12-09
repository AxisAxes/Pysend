
def build_model_tree(dataset_path, features_list, target, max_leaf_nodes):
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor

    
    df = pd.read_csv(dataset_path)
    df.head()
    y = df[target]
    features = list(features_list)
    X = df[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    dtree_model1 = DecisionTreeRegressor(random_state=1)
    dtree_model1.fit(train_X, train_y)

    val_predictions1 = dtree_model1.predict(val_X)
    val_mae1 = mean_absolute_error(val_predictions1, val_y)
    print('Validation MAE when not specifying max_leaf_nodes: {:,.0f}'.format(val_mae1))

    dtree_model2 = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    dtree_model2.fit(train_X, train_y)
    val_predictions2 = dtree_model2.predict(val_X)
    val_mae = mean_absolute_error(val_predictions2, val_y)
    print('Validation MAE for best value of max_leaf_nodes: {:,.0f}'.format(val_mae))
