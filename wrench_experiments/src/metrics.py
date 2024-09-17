from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def exp_eval(y_true, y_pred, a=None, fairness=True, cond=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    result = {}
    
    if cond is not None:
        result['condition'] = cond
        
    result["accuracy"] = accuracy
    result["fscore"] = fscore
    result["precision"] = precision
    result["recall"] = recall
        
    return result