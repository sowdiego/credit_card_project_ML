from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt  
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

st.set_page_config(layout="wide")

    
#### Partie A : Exploration de notre Dataset
creditcard_data=pd.read_csv('creditcard.csv')
acc = []
score = []
recall = []
score_f1 = []
st.title('Détection de Fraude par Carte de Crédit')
st.write("-----")
# taches=['1. Exploration de données', '2. Prétraitement des Données', '3. Modélisation', '4. Évaluation du Modèle', '5. Optimisation', '6. Interprétation', '7. Rapport Final']

# choix=st.checkbox('Selectionner une activité:', taches)


if st.sidebar.checkbox('1. Exploration de données'):
    st.subheader('1. Exploration de données')
    st.write("-----")
    slider_ui= st.slider('''choisir l'intervalle de valeurs à afficher''', 1, creditcard_data.shape[0], (50000,200000))
    st.dataframe(creditcard_data.iloc[list(slider_ui)[0]: list(slider_ui)[1]])

    st.write(f'Shape:  { creditcard_data.shape }')
   

    st.write("Describe: ")
    st.write(creditcard_data.describe())

    st.write("Valeur null: ")
    st.write(creditcard_data.isnull().sum())



if st.sidebar.checkbox('2. Prétraitement des Données'):
    st.subheader('2. Prétraitement des Données')
    st.write("""
                - Normalisons les montant avec StandardScaler
                """)
    
    st.write("""
                L'idée derrière la normalisation est de mettre toutes les caractéristiques (features) à la même échelle, généralement en les centrant autour de zéro et en ajustant leur dispersion.
             """)
    
    sc = StandardScaler()
    creditcard_data['Amount'] = sc.fit_transform(pd.DataFrame(creditcard_data['Amount']))
    st.write(creditcard_data.head())

    st.write("""
               - Suppression des doublons ...
                """)
    creditcard_data = creditcard_data.drop_duplicates()
    st.write(f'Nouveau Shape:  { creditcard_data.shape }')

    st.write("""
               - Distibution de notre variable target
                """)
    st.write(creditcard_data['Class'].value_counts())
    
    st.write("-----")

if st.sidebar.checkbox('3. Modélisation'):
    st.subheader('3. Modélisation & Optimisation')

    st.write("""
            - Sous-échantillonnage
            """)
   
    st.write(""" Logiquement pour que les modeles soient bien entrainées il serait préférable d'avoir le méme nombre de classe target.
             Du coup on va creer un nouveau dataframe df_echantillonnage qui aura 473 lignes avec la classe 1 et 473 lignes avec la classe 0.
             """)
    normal = creditcard_data[creditcard_data['Class']==0]
    fraud = creditcard_data[creditcard_data['Class']==1]

    normal_sample = normal.sample(n=473)

    df_echantillonnage = pd.concat([normal_sample, fraud],ignore_index=True)
    st.write(df_echantillonnage.head())
    st.write(f'Shape:  { df_echantillonnage.shape }')
    st.write(df_echantillonnage['Class'].value_counts())


    X = df_echantillonnage.drop('Class', axis=1)
    y = df_echantillonnage['Class']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

    modeles = ["1. Logistic Regression", "2. Decision Tree Classifier", "3. Random Forest Classifier"]
    #choix = st.sidebar.selectbox("Selectionner un un modele:", modeles)
    st.write("1. Logistic Regression")
    st.write("""
            Le modèle de régression logistique utilise une fonction logistique pour estimer la probabilité qu'une observation appartienne à une classe particulière. Cette fonction prend en compte une combinaison linéaire des variables d'entrée, à laquelle est appliquée la fonction logistique (ou sigmoïde).
            """)
    log = LogisticRegression()
    log.fit(X_train, y_train)

    #prediction
    y_predict1 = log.predict(X_test)
    #Accurancy
    acc1 = accuracy_score(y_test, y_predict1)
    st.write("Accuracy: ", round(acc1, 2))
    acc.append(acc1*100)
    #score
    score1 = precision_score(y_test, y_predict1)
    score.append(score1)

    recall1 = recall_score(y_test, y_predict1)
    recall.append(recall1)
    
    scoref1_1 = f1_score(y_test, y_predict1)
    score_f1.append(scoref1_1)

    # Validation croisée pour évaluer la généralisation du modèle
    cv1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores1 = cross_val_score(log, X, y, cv=cv1, scoring='accuracy')

    # Courbe ROC
    fpr1, tpr1, _ = roc_curve(y_test, log.predict_proba(X_test)[:, 1])
    roc_auc1 = auc(fpr1, tpr1)

    st.write("Précision: ", round(score1, 2))
    st.write("Rappel: ", round(recall1, 2))
    st.write("F1-score: ", round(scoref1_1, 2))

    st.write("2. Decision Tree Classifier")
    st.write("""
                Les arbres de décision fonctionnent en divisant récursivement l'ensemble de données en sous-ensembles plus petits, basés sur des critères spécifiques, jusqu'à ce que chaque sous-ensemble soit suffisamment homogène en termes de classe (dans le cas de la classification) ou de valeur de sortie (dans le cas de la régression).
                """)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    #prediction
    y_predict2 = dt.predict(X_test)
    #Accurancy
    acc2 = accuracy_score(y_test,y_predict2)
    st.write("Accuracy: ", round(acc2, 2))
    acc.append(acc2*100)
    #score
    score2 = precision_score(y_test, y_predict2)
    score.append(score2)

    recall2 = recall_score(y_test, y_predict2)
    recall.append(recall2)

    scoref1_2 = f1_score(y_test, y_predict2)
    score_f1.append(scoref1_2)

    # Validation croisée pour évaluer la généralisation du modèle
    cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores2 = cross_val_score(dt, X, y, cv=cv2, scoring='accuracy')

    # Courbe ROC
    fpr2, tpr2, _ = roc_curve(y_test, dt.predict_proba(X_test)[:, 1])
    roc_auc2 = auc(fpr2, tpr2)

    st.write("Précision: ", round(score2, 2))
    st.write("Rappel: ", round(recall2, 2))
    st.write("F1-score: ", round(scoref1_2, 2))

    st.write("3. Random Forest Classifier")
    st.write("""
                C’est un algorithme qui se base sur l’assemblage d’arbres de décision. Il est assez intuitif à comprendre, rapide à entraîner et il produit des résultats généralisables.
                """)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    #prediction
    y_predict3 = rf.predict(X_test)
    #Accurancy
    acc3 = accuracy_score(y_test, y_predict3)
    st.write("Accuracy: ", round(acc3, 2))
    acc.append(acc3*100)
    #score
    score3 = precision_score(y_test, y_predict3)
    score.append(score3)

    recall3 = recall_score(y_test, y_predict3)
    recall.append(recall3)

    scoref1_3 = f1_score(y_test, y_predict3)
    score_f1.append(scoref1_3)

    # Validation croisée pour évaluer la généralisation du modèle
    cv3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores3 = cross_val_score(rf, X, y, cv=cv3, scoring='accuracy')

    # Courbe ROC
    fpr3, tpr3, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
    roc_auc3 = auc(fpr3, tpr3)

    st.write("Précision: ", round(score3, 2))
    st.write("Rappel: ", round(recall3, 2))
    st.write("F1-score: ", round(scoref1_3, 2))
    st.write("-----")

if st.sidebar.checkbox('4. Évaluation du Modèle'):
    st.subheader('4. Évaluation du Modèle')
    acc_df = pd.DataFrame({'Models': ['LR', 'DT', 'RF'], "ACC": acc})
    score_df = pd.DataFrame({'Models': ['LR', 'DT', 'RF'], "SCORE": score})
    recall_df = pd.DataFrame({'Models': ['LR', 'DT', 'RF'], "RECALL": recall})
    score_f1_df = pd.DataFrame({'Models': ['LR', 'DT', 'RF'], "score_f1": score_f1})
    left_column,center_column1, center_column2,  right_column = st.columns(4)   

    with left_column:
        fig_top_clients = plt.figure(figsize=(4, 2))
        sns.barplot(x=acc_df['Models'], y=acc_df['ACC'])
        plt.title('ACCURACY')
        plt.xticks(rotation=45, ha='center')
        st.pyplot(fig_top_clients)

    with center_column1:
        fig_top_clients = plt.figure(figsize=(4, 2))
        sns.barplot(x=score_df['Models'], y=score_df['SCORE'])
        plt.title('SCORE PRECISION')
        plt.xticks(rotation=45, ha='center')
        st.pyplot(fig_top_clients)
    
    with center_column2:
        fig_top_clients = plt.figure(figsize=(4, 2))
        sns.barplot(x=recall_df['Models'], y=recall_df['RECALL'])
        plt.title('RECALL SCORE')
        plt.xticks(rotation=45, ha='center')
        st.pyplot(fig_top_clients)
        
    with right_column:
        fig_top_clients = plt.figure(figsize=(4, 2))
        sns.barplot(x=recall_df['Models'], y=score_f1_df['score_f1'])
        plt.title('F1 SCORE')
        plt.xticks(rotation=45, ha='center')
        st.pyplot(fig_top_clients)

    st.subheader('Validation Croisée pour Généralisation')
    left_column, center_column,  right_column = st.columns(3)   

   

    with left_column:
        # Afficher la validation croisée
        st.write(f'Précision moyenne Logistic Regression : {round(np.mean(cross_val_scores1), 2)}')

        # Afficher la courbe ROC
        st.write('Courbe ROC Logistic Regression')
        st.write(f'Aire : {round(roc_auc1, 2)}')
        st.line_chart(pd.DataFrame({'False Positive Rate': fpr1, 'True Positive Rate': tpr1}).set_index('False Positive Rate'))

    with center_column:
        # Afficher la validation croisée
        st.write(f'Précision moyenne Tree Classifier: {round(np.mean(cross_val_scores2), 2)}')

        # Afficher la courbe ROC
        st.write('Courbe ROC Tree Classifier')
        st.write(f'Aire : {round(roc_auc2, 2)}')
        st.line_chart(pd.DataFrame({'False Positive Rate': fpr2, 'True Positive Rate': tpr2}).set_index('False Positive Rate'))
       
        
    with right_column:
        # Afficher la validation croisée
        st.write(f'Précision moyenne Random Forest: {round(np.mean(cross_val_scores3), 2)}')

        # Afficher la courbe ROC
        st.write('Courbe ROC Random Forest')
        st.write(f'Aire : {round(roc_auc3, 2)}')
        st.line_chart(pd.DataFrame({'False Positive Rate': fpr3, 'True Positive Rate': tpr3}).set_index('False Positive Rate'))
        

if st.sidebar.checkbox('5. Optimisation '):
    st.subheader('5. Optimisation')
   
    choix = st.sidebar.selectbox("Selectionner un modele", modeles)
    if choix == '1. Logistic Regression':
        st.subheader('Optimisation du modele 1')
        # Ajustement des hyperparamètres
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        st.write("Avec grid_search on essaye d'ajuster les meilleurs hyperparametres ")
        st.write('grid search params:', param_grid)
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Meilleurs hyperparamètres
        best_params = grid_search.best_params_
        st.write(f'Meilleurs hyperparamètres : {best_params}')

        # Modèle avec meilleurs hyperparamètres
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Évaluation du modèle avec meilleurs hyperparamètres
        y_pred_best = best_model.predict(X_test)
        precision_best = precision_score(y_test, y_pred_best)
        recall_best = recall_score(y_test, y_pred_best)
        f1_best = f1_score(y_test, y_pred_best)

        # Afficher les nouvelles mesures de performance
        st.write(f'Précision : {round(precision_best, 2)}')
        st.write(f'Rappel : {round(recall_best, 2)}')
        st.write(f'F1-score : {round(f1_best, 2)}')


       
    
    if choix == '2. Decision Tree Classifier':
        st.subheader('Optimisation du modele 2')
        # Ajustement des hyperparamètres pour le modèle d'arbre de décision
        param_grid_dt = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        st.write("Avec grid_search on essaye d'ajuster les meilleurs hyperparametres ")
        st.write('grid search params:', param_grid_dt)

        grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
        grid_search_dt.fit(X_train, y_train)

        # Meilleurs hyperparamètres pour le modèle d'arbre de décision
        best_params_dt = grid_search_dt.best_params_
        st.write(f'Meilleurs hyperparamètres : {best_params_dt}')

        # Modèle d'arbre de décision avec meilleurs hyperparamètres
        best_model_dt = grid_search_dt.best_estimator_
        best_model_dt.fit(X_train, y_train)

        # Évaluation du modèle d'arbre de décision avec meilleurs hyperparamètres
        y_pred_dt_best = best_model_dt.predict(X_test)
        precision_dt_best = precision_score(y_test, y_pred_dt_best)
        recall_dt_best = recall_score(y_test, y_pred_dt_best)
        f1_dt_best = f1_score(y_test, y_pred_dt_best)

        # Afficher les nouvelles mesures de performance pour le modèle d'arbre de décision
        st.write(f'Précision : {round(precision_dt_best, 2)}')
        st.write(f'Rappel : {round(recall_dt_best, 2)}')
        st.write(f'F1-score : {round(f1_dt_best, 2)}')

       

    
    if choix == '3. Random Forest Classifier':
        st.subheader('Optimisation du modele 3')
        

         # Ajustement des hyperparamètres pour un modèle de forêt aléatoire
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        st.write("Avec grid_search on essaye d'ajuster les meilleurs hyperparametres ")
        st.write('grid search params:', param_grid_rf)

        grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
        grid_search_rf.fit(X_train, y_train)

        # Meilleurs hyperparamètres pour la forêt aléatoire
        best_params_rf = grid_search_rf.best_params_
       
        st.write(f'Meilleurs hyperparamètres : {best_params_rf}')

        # Modèle de forêt aléatoire avec meilleurs hyperparamètres
        best_model_rf = grid_search_rf.best_estimator_
        best_model_rf.fit(X_train, y_train)

        # Évaluation du modèle de forêt aléatoire avec meilleurs hyperparamètres
        y_pred_rf_best = best_model_rf.predict(X_test)
        precision_rf_best = precision_score(y_test, y_pred_rf_best)
        recall_rf_best = recall_score(y_test, y_pred_rf_best)
        f1_rf_best = f1_score(y_test, y_pred_rf_best)

        # Afficher les nouvelles mesures de performance pour la forêt aléatoire
        st.write(f'Précision : {round(precision_rf_best, 2)}')
        st.write(f'Rappel : {round(recall_rf_best, 2)}')
        st.write(f'F1-score : {round(f1_rf_best, 2)}')

      


if st.sidebar.checkbox('6. Interprétation'):
    st.subheader('6. Interprétation')
    st.write("""
            Apres évaluation de l'ensemble des modéles il apparait que les modeles de Regression et de Random forest donnent des resultats plus optimals en termes de qualité de classification
             avec des précision de 98% et 99% aussi la validation croisée et la courbe ROC montrent qu'ils ont respectivement des précision et une aire plus
             considérable.
             
            """)


