import pandas as pd
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score

def ripperIS(X_train, X_test, y_train, y_test):
    
    # Wittgenstein uses fit-predict-score syntax that’s similar to scikit-learn’s.
    # We’ll train a RIPPER classifier, with the positive class defined as 1 (easy).
    
    clf = lw.RIPPER(random_state=42)
    clf.fit(X_train, y_train, pos_class=1)
    
    # Collect performance metrics
    precision = clf.score(X_test, y_test, precision_score)
    recall = clf.score(X_test, y_test, recall_score)
    cond_count = clf.ruleset_.count_conds()
    
    print(f'precision: {precision} recall: {recall} conds: {cond_count}')
    
    # We can access our trained model using the clf.ruleset_ attribute. 
    # A trained ruleset model represents a list of “and’s” of “or’s”:
    #clf.ruleset_.out_pretty()
    ruleset = clf.ruleset_
    
    return [precision, recall, cond_count], ruleset