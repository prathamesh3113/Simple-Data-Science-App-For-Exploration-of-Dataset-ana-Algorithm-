import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")
# adding color as green
st.markdown(
    """                                                                     
    <style>                                                                 
    .sidebar .sidebar-content {                                             
        background-image: linear-gradient(#84d162,#2c8530);                 
        color: white;                                                       
    }                                                                       
    </style>                                                                
    """, unsafe_allow_html=True,
)


def main():
    """Simple Data Science App"""
    st.markdown("<h1 style='text-align: left; color: black;'>Simple Data Science App</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: black;'>Aim: Explore different classifier and datasets using Machine learning</h2>", unsafe_allow_html=True)


    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.write("")
        image = Image.open('C:/Users/Vikas/PycharmProject/CreditCardDetection/creditcrack1.png')
        st.image(image, caption='', use_column_width=True)

    elif choice == "Login":

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                st.markdown("<h2 style='text-align: left; color: black;'> Select Dataset </h2>", unsafe_allow_html=True)
                dataset_name = st.selectbox("",["Iris", "Breast Cancer", "Wine"])
                st.markdown("<h2 style='text-align: left; color: black;'>Select Classifier </h2>", unsafe_allow_html=True)
                classifier_name = st.selectbox('', ('KNN', 'SVM', 'Random Forest'))
                st.markdown("<h3 style='text-align: left; color: black;'>Dataset:</h3>", unsafe_allow_html=True)
                st.write(f"## {dataset_name}")
                def get_dataset(name):
                    data = None
                    if name == 'Iris':
                        data = datasets.load_iris()
                    elif name == 'Wine':
                        data = datasets.load_wine()
                    else:
                        data = datasets.load_breast_cancer()
                    X = data.data
                    y = data.target
                    return X, y

                X, y = get_dataset(dataset_name)
                st.markdown("<h3 style='text-align: left; color: black;'>Shape of dataset:</h3>",unsafe_allow_html=True)
                st.write(X.shape)
                st.markdown("<h3 style='text-align: left; color: black;'>Number of classes:</h3>",unsafe_allow_html=True)
                st.write(len(np.unique(y)))

                def add_parameter_ui(clf_name):
                    params = dict()
                    if clf_name == 'SVM':
                        C = st.slider('C', 0.01, 10.0)
                        params['C'] = C
                    elif clf_name == 'KNN':
                        K = st.slider('K', 1, 15)
                        params['K'] = K
                    else:
                        max_depth = st.slider('max_depth', 2, 15)
                        params['max_depth'] = max_depth
                        n_estimators = st.slider('n_estimators', 1, 100)
                        params['n_estimators'] = n_estimators
                    return params

                params = add_parameter_ui(classifier_name)

                def get_classifier(clf_name, params):
                    clf = None
                    if clf_name == 'SVM':
                        clf = SVC(C=params['C'])
                    elif clf_name == 'KNN':
                        clf = KNeighborsClassifier(n_neighbors=params['K'])
                    else:
                        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                                           max_depth=params['max_depth'], random_state=1234)
                    return clf

                clf = get_classifier(classifier_name, params)
                #### CLASSIFICATION ####

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.markdown("<h3 style='text-align: left; color: black;'>Classifier:</h3>", unsafe_allow_html=True)
                st.write(f"{classifier_name}")
                st.markdown("<h3 style='text-align: left; color: black;'>Accuracy:</h3>", unsafe_allow_html=True)
                st.write(acc)

                #### PLOT DATASET ####
                # Project the data onto the 2 primary principal components
                st.markdown("<h3 style='text-align: left; color: black;'>Visualization of Given Dataset</h3>",unsafe_allow_html=True)
                pca = PCA(2)
                X_projected = pca.fit_transform(X)

                x1 = X_projected[:, 0]
                x2 = X_projected[:, 1]

                fig = plt.figure()
                plt.scatter(x1, x2,
                            c=y, alpha=0.8,
                            cmap='viridis')

                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.colorbar()

                # plt.show()
                st.pyplot(fig)

                st.markdown("<h3 style='text-align: left; color: black;'>Check User Detalis</h3>",unsafe_allow_html=True)
                task = st.selectbox('', ['Conclusion','User Profiles'])
                if task =="Conclusion":

                    st.markdown("<h3 style='text-align: left; color: black;'>Interactive Web App with Streamlit and"
                                " Scikit-learn and it explore different dataset and algo</h3>",
                                unsafe_allow_html=True)

                elif task == "User Profiles":
                    st.markdown("<h3 style='text-align: left; color: black;'>User Profiles Of App</h3>",unsafe_allow_html=True)
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                    st.dataframe(clean_db)


            else:
                st.warning("Incorrect Username/Password")





    elif choice == "SignUp":
        st.markdown("<h2 style='text-align: left; color: black;'> Create New Account </h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; color: black;'> Username </h3>", unsafe_allow_html=True)
        new_user = st.text_input("")
        st.markdown("<h3 style='text-align: left; color: black;'> Password </h3>", unsafe_allow_html=True)
        new_password = st.text_input("Password",type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")



if __name__ == '__main__':
    main()