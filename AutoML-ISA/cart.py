from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score

def cartIS(X_train, X_test, y_train, y_test):
    
    tree = DecisionTreeClassifier(random_state = 42, max_depth = 3, min_samples_split = 30)
    tree = tree.fit(X_train, y_train)
    
    # Predict test data
    y_pred = tree.predict(X_test)
    
    # Collect performance metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    

    print(f'precision: {precision} recall: {recall}')
    
    return [precision, recall], tree