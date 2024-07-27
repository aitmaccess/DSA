import pandas as pd 

df = pd.read_csv('/content/ML LAB/enjoysport.csv') 
a = df.values.tolist() 

print(df) 

n = len(a[0]) - 1
print("\nThe initial value of hypothesis:") 

s = ['0'] * n 
g = ['?'] * n 

print("\nThe most specific hypothesis S0:", s) 
print("\nThe most general hypothesis G0:", g) 

s = a[0][:-1] 
temp = [] 

print("\nCandidate Elimination algorithm\n") 

for i in range(len(a)): 
    if a[i][n] == "yes":
        for j in range(n): 
            if a[i][j] != s[j]: 
                s[j] = '?'
        for j in range(n): 
            for k in range(len(temp)): 
                if temp[k][j] != '?' and temp[k][j] != s[j]: 
                    del temp[k]
    if a[i][n] == "no":
        for j in range(n): 
            if s[j] != a[i][j] and s[j] != '?': 
                g[j] = s[j]
        if g not in temp: 
            temp.append(g) 
        g = ['?'] * n 

    print("\nFor Training Example No :{0} the hypothesis is S{0}".format(i + 1), s) 
    if len(temp) == 0: 
        print("For Training Example No :{0} the hypothesis is G{0}".format(i + 1), g) 
    else: 
        print("For Training Example No :{0} the hypothesis is G{0}".format(i + 1), temp)
