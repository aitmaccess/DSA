!pip install pgmpy 

import pandas as pd
from pgmpy.models import BayesianModel 
from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.inference import VariableElimination 

data = pd.read_csv('datasetheart.csv', names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'RESULT']) 

print(data.head(5)) 
print(data.tail(5)) 

model = BayesianModel([("A", "B"), ("B", "C"), ("C", "D"), ("D", "RESULT")]) 
model.fit(data, estimator=MaximumLikelihoodEstimator) 

infer = VariableElimination(model) 
q = infer.query(variables=['RESULT'], evidence={"C": 2}) 
print(q)
