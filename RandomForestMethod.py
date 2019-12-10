

def build_model_forest(dataset_path, features_list, target, max_leaf_nodes, max_depth=None, n_estimators=None):
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, plot_tree

    def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return mean_absolute_error(y_v, preds)

    df = pd.read_csv(dataset_path)
    df.head()
    y = df[target]
    features = list(features_list)
    X = df[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    rf_model1 = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators )
    rf_model1.fit(train_X, train_y)
    rf_model_predict1 = rf_model1.predict(val_X)

    rf_val_mae1 = mean_absolute_error(val_y, rf_model_predict1)

    print('Validation MAE for Random Forest Model: {:,.0f}'.format(rf_val_mae1))

    rf_model2 = RandomForestRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,  n_estimators=n_estimators )
    rf_model2.fit(train_X, train_y)
    rf_model_predict2 = rf_model2.predict(val_X)

    rf_val_mae2 = mean_absolute_error(val_y, rf_model_predict2)

    print('Validation MAE for Random Forest Model with max_leaf_nodes: {:,.0f}'.format(rf_val_mae2))

    plot1 = plot_tree(rf_model1)
    plot2 = plot_tree(rf_model2)

    a = [rf_model1, rf_model1 ]
    list_of_model_maes = [score_model(i) for i in a ]
    best_model = a[list_of_model_maes.index(min(list_of_model_maes))]

    print('This is the best model : {}'.format(best_model))
        
    print('Plot of model tree (not specifying max_leaf_nodes)\n\n{} '.format(plot1))
    print('Plot of the model for best value of max_leaf_nodes\n\n{}'.format(plot2))


