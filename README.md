# Gradient Descent for L2-regularized Logistic Regression

This code implements the gradient descent algorithm to solve L2-regularized logistic regression. 

See the gradient_descent_simulated.py file for an example of applying gradient descent algorithm to solve L2-regularized logistic regression problem on simulated data. To run this example, simply load packages and run the gradient_descent_simulated.py file using Python:

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
python gradient_descent_simulated.py
```

The output will look like this:



See the gradient_descent_real_world.py file for an example of applying gradient descent algorithm to solve L2-regularized logistic regression problem on a real world data, which is the the "Spam" data in spam.csv file. It can also be downloaded from https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data. To run this example, simply load packages and run the gradient_descent_real_world.py file using Python:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
python gradient_descent_real_world.py
```

See the comparison.py file for an example of comparing the results from gradient descent's implementation and scikit-learn's on "spam" dataset. To run this example, simply load packages and run the gcomparison.py file using Python:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
python comparison.py
```
