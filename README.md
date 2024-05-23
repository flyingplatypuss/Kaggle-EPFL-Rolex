# 👋 Kaggle_EPFL_Rolex
Hail and welcome, we are team Rolex (Johann Gremaud and Anna Perrottet).

We have been working on building a classification model that can identify the difficulty(A1-C2) of any french text. 

$\text{\LARGE{\color{red}{Models}}}$

We have used 5 different models: 
- Logistic Regression, 
- K Nearest Neighbours
- Decision Tree
- Random Forest
- Large Language Model (LLM) with camemBERT

$\text{\Large{\color{#EB5406}{Performances}}}$

<img src = "https://github.com/flyingplatypuss/Kaggle-EPFL-Rolex/assets/146196573/ef66f7f3-9ab0-4df9-babe-fbc918cd69a2" width = "500" height="auto"/>

Our most important metric is accuracy. KNN, Random Forest and Decision Tree have around 40% of accuracy. Logistic regression performs better with 47%. 
this can be explained by several factors: 
- the size of the dataset
- the complexity of the classifier e.g. camembert is able to "understand" the meaning of the sentence
  
$\text{\LARGE{\color{#00008B}{LLM with camemBERT}}}$

$\text{\Large{\color{blue}{Confusion Matrix}}}$

<img src="https://github.com/flyingplatypuss/Kaggle-EPFL-Rolex/assets/146196573/0261567a-3c97-43e5-b169-d3e865d11d8c" alt="WhatsApp Image 2024-05-21 à 21 41 23_3f497c78" width="500" height="auto"/>

By retraining the CamemBERT model on the full training dataset we achieve an accuracy of 0.587. By running it many times and by taking the most represented difficulty level for each sentence, we achieved an accuracy of 0.62, placing ourselves in the 3rd position of the Kaggle competition.

$\text{\Large{\color{blue}{Limitations and flaws}}}$

The errors from the model come from:
- the limited size of the training dataset (n = 1200?)
- camemBERT is simpler than BERT but also less performant
- the neural network operates according to its own logic, which is not always aligned with human neural network



### 🥐 Follow this link to experience a very french journey through Large Language Models:
[This way!](https://youtu.be/xTXCNCszG50)

### ☕ Pigeons will come for you if you leave crumbs, but those from Paris may give you more, try your new favourite french learning app:
[Do it yourself](https://pigeons-and-crumbs.streamlit.app/)

