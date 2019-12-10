def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

def build_model_tree(dataset_path, features_list, target, max_leaf_nodes):
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor, plot_tree


    
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

    plot1 = plot_tree(dtree_model1)
    plot2 = plot_tree(dtree_model2)

    a = [dtree_model1, dtree_model2 ]
    list_of_model_maes = [score_model(i) for i in a ]
    best_model = a[list_of_model_maes.index(min(list_of_model_maes))]

    print('This is the best model : {}'.format(best_model))


    print('Plot of model tree (not specifying max_leaf_nodes)\n\n{} '.format(plot1))
    print('Plot of the model for best value of max_leaf_nodes\n\n{}'.format(plot2))
