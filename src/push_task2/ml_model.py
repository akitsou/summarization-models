from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

import datetime
#%%
start = datetime.datetime.now()


X_train_df = pd.read_csv('sample_trainset1000.csv')
X_test_full_df = pd.read_csv('sample_testset300 (1).csv')

#%%
#in case there are Nan values in the column, fill them with 0s
X_train_df['max_cosine_similarity'] = X_train_df['max_cosine_similarity'].fillna(0.000000)
# Replace NaN values with 0.0
X_test_full_df['max_cosine_similarity'] = X_test_full_df['max_cosine_similarity'].fillna(0.000000)
#%%
y_train_df = pd.DataFrame(X_train_df['label'],columns=["label"])
X_train_df = X_train_df.drop(columns=['label'])

#%%
X_test_ids = X_test_full_df['new_article_sentence'].tolist()
X_test_articles = [x.split("-")[0] for x in X_test_ids]

#%%
y_test_df = pd.DataFrame(X_test_full_df['label'],columns=["label"])
X_test_df = X_test_full_df.drop(columns=['label','new_article_sentence'])


print(X_train_df.head(10))
print(y_train_df)
#%%
y_train_np = np.array(y_train_df).squeeze()
X_train_np = np.array(X_train_df)

y_test_np = np.array(y_test_df).squeeze()
X_test_np = np.array(X_test_df)
#%%
# clf = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100, 100, 100), max_iter=1000, shuffle=True)
clf = AdaBoostClassifier(n_estimators=20, algorithm='SAMME.R')
#clf = LogisticRegression()
#clf = RandomForestRegressor(n_estimators=20, max_depth=5)
#clf = RandomForestClassifier(max_depth=4, random_state=0)
#clf = svm.SVR(kernel='linear', C=1e3, gamma=0.1)

scaler = MinMaxScaler()
X_train_np = scaler.fit_transform(X_train_np)

clf.fit(X_train_np, y_train_np)


X_test_np = scaler.transform(X_test_np)
value = clf.predict(X_test_np)
#value = clf.predict_proba(X_test_np)

#value = value[:,0]

#%%
value = value.tolist()

final_df = pd.DataFrame(
    {'article': X_test_articles,
    'index': X_test_ids,
     'score': value
    }).set_index('index', drop=True)

# print('final_df')
# print(final_df)
#%%
#Choose the 3 top scoring sentences in asceding order
# output_df = final_df.sort_values(['article','sentence','score'], ascending=[1,0]).groupby('article').head(3)

output_df = final_df.sort_values(['article','score'], ascending=[1,0]).groupby('article').head(3)

# print('output_df')
# print(output_df)

selected_sentences = output_df.index.values
selected_sentences = selected_sentences.tolist()

# print('selected_sentences')
# print(selected_sentences)
#%%
y_pred = [1 if x in selected_sentences else 0 for x in X_test_ids]

y_pred_df = pd.DataFrame(
    {'y_pred': y_pred,
    })

# print('y_pred_df')
# print(y_pred_df.head(20))

#%%
y_pred_df.to_csv('predictions.csv')
