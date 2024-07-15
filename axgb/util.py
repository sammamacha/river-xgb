import numpy as np 
import pandas as pd

def row_convert(x) -> np.array:
    """
    row_convert returns a numpy array given dictionary/pd.Series input to prevent 
    hassle--as the default learn_one parameter takes a dict() and .iloc[] by 
    default returns a pd.Series. 
    """

    if isinstance(x, dict): 
        x = np.array(list(x.values()))
    elif isinstance(x, pd.Series): 
        x = x.to_numpy()
    elif isinstance(x, list): 
        x = np.array(x)
    # Not sure what else would be passed, just return
    return x 

def matrix_convert(x) -> np.ndarray: 
    """
    matrix_convert 
    pd.Dataframe -> np.ndarray  
    """
    if isinstance(x, pd.DataFrame): 
        x = x.values() 
    return x 