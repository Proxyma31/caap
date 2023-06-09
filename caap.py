import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.tree import export_graphviz
from xgboost import XGBClassifier
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from statistics import mode

im = Image.open('icon.png')
st.set_page_config(layout="centered", page_title="CAAp", page_icon=im)
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
with st.sidebar:
    st.title(":mag: Navigation Bar")
    s = option_menu(
        menu_title=None,
        options=["Home", "Project", "Devs", "Future Scope", "Help"],
        icons=["house", "book", "robot", "clock", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link-selected": {"background-color": "green"}
        }
    )

if s == "Home":
    st.title(":green[CAAp]")
    st.subheader("An Application For Concentration Analysis Powered By Machine Learning")
    st.header(":microscope:")

if s == "Project":
    st.title('Concentration of Inositol Analysis')
    s1 = option_menu(
        menu_title=None,
        options=["Data and Model"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "icon": {"color": "white", "font-size": "25px"},
            "nav-link-selected": {"background-color": "violate"}
        }
    )
    if s1 == "Data and Model":
        uploaded_files = st.file_uploader("Choose file for Training in xlsx Format", type="xlsx")
        if uploaded_files:
            df = pd.read_excel(uploaded_files, engine="openpyxl")
            st.header(':white_check_mark: Uploaded')
            X = df.iloc[:, 0:10]
            Y = df.iloc[:, -1]
            sc = MinMaxScaler()
            sc.fit(X)
            X_scaled = sc.transform(X)

            img_cluster = Image.open("img_new.png")

            choose = st.selectbox("Select a option", ('None', 'Dataframe', 'Plot', 'Visualize Dataframe',
                                                      'Number of Distinct Concentrations'))
            if choose == 'Dataframe':
                st.text('Dataframe')
                st.dataframe(df.style.format("{:.11%}"))
            elif choose == 'Number of Distinct Concentrations':
                st.text(Y.nunique())
            elif choose == 'None':
                pass
            elif choose == 'Visualize Dataframe':
                st.image(img_cluster, caption="Features vs Classes")
            else:
                plt.figure(figsize=[22, 12])
                plt.subplot(2, 5, 1)
                plt.scatter(df['c1'], df['con'])
                plt.xlabel('Current Column 1')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 2)
                plt.scatter(df['c2'], df['con'])
                plt.xlabel('Current Column 2')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 3)
                plt.scatter(df['c3'], df['con'])
                plt.xlabel('Current Column 3')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 4)
                plt.scatter(df['c4'], df['con'])
                plt.xlabel('Current Column 4')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 5)
                plt.scatter(df['c5'], df['con'])
                plt.xlabel('Current Column 5')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 6)
                plt.scatter(df['c6'], df['con'])
                plt.xlabel('Current Column 6')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 7)
                plt.scatter(df['c7'], df['con'])
                plt.xlabel('Current Column 7')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 8)
                plt.scatter(df['c8'], df['con'])
                plt.xlabel('Current Column 8')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 9)
                plt.scatter(df['c9'], df['con'])
                plt.xlabel('Current Column 9')
                plt.ylabel('Concentration')
                plt.subplot(2, 5, 10)
                plt.scatter(df['c10'], df['con'])
                plt.xlabel('Current Column 10')
                plt.ylabel('Concentration')
                st.pyplot(plt)

        if s1 == "Data and Model":
            uploaded_files_Custom = st.file_uploader("Choose Custom file for testing in xlsx Format", type="xlsx")
            if uploaded_files_Custom:
                df = pd.read_excel(uploaded_files_Custom, engine="openpyxl")
                st.header(':white_check_mark: Uploaded')
                X_custom = df.iloc[:, 0:10]
                Y_custom = df.iloc[:, -1]
                sc1 = MinMaxScaler()
                sc1.fit(X_custom)
                X_scaled_custom = sc1.transform(X_custom)
        st.header("Select a Model")
        st.subheader(":pushpin: Note from the Devs, Support Vector Machine, K Nearest Neighbour is recommended for "
                     "there better accuracy and Xgboost is not recommended due to its time costing processing "
                     ":pushpin:")
        button1 = st.checkbox('Decision Tree Classifier')
        if button1:
            x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=50)

            model = DecisionTreeClassifier(max_depth=100, max_leaf_nodes=500, random_state=40)
            model.fit(x_train, y_train)
            model.score(x_test, y_test)
            y_predicted_dct: object = model.predict(x_test)

            c2 = confusion_matrix(y_test, y_predicted_dct)

            accuracy = f1_score(y_test, y_predicted_dct, average='weighted')

            y_predicted_dct_custom = model.predict(X_scaled_custom)

            dot = tree.export_graphviz(model,
                                       filled=True,
                                       feature_names=('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'),
                                       class_names=('50', '80', '100', '200', '300', '400'),
                                       label="all")

            choose_dct1 = st.button('Model Accuracy')
            if choose_dct1:
                st.header('Model Accuracy')
                st.subheader(accuracy * 100)
            else:
                pass
            choose_dct2 = st.button('Heatmap')
            if choose_dct2:
                plt.figure(figsize=(8, 6))
                sns.heatmap(c2, annot=True)
                plt.xlabel('predict')
                plt.ylabel('Truth')
                st.pyplot(plt)
            else:
                pass
            choose_dct3 = st.button('Plot')
            if choose_dct3:
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(y_test.index, y_test, color='black')
                plt.xlabel('index')
                plt.ylabel('Test_data')
                plt.subplot(1, 2, 2)
                plt.scatter(y_test.index, y_predicted_dct, color='green')
                plt.xlabel('index')
                plt.ylabel('Predicted_data')
                st.pyplot(plt)
            else:
                pass
            choose_dct4 = st.button('Visualize Model')
            if choose_dct4:
                st.graphviz_chart(dot)
            else:
                pass
            choose_dct5 = st.button('Custom Data Input Result')
            if choose_dct5:
                st.header("Custom output data")
                for i in range(0, len(y_predicted_dct_custom)):
                    st.subheader(
                        f"Input No.  {i+1}  the model predicted Concentration  -->  {y_predicted_dct_custom[i]}")
            else:
                pass
            choose_dct6 = st.button('Suggestion on Custom Data')
            if choose_dct6:
                ans = mode(y_predicted_dct_custom)
                sugg = ((ans / 1000000) / 180.16) * 1000000
                sugg1 = sugg * 1000
                st.subheader(f"In the Custom Input The Most Frequent Occurring Concentration is {ans} microMolar")
                st.subheader(f"Which is Equal to {sugg1} mg/L")
                if 225 < sugg1 < 1500:
                    st.subheader("Because the Current Concentration is in Ideal Range of 225 to 1500 mg/L according "
                                 "to N.L.H.")
                    st.subheader("So it is Recommended :thumbsup:")
                else:
                    st.subheader("Because the Current Concentration is not in Ideal Range of 225 to 1500 mg/L according"
                                 " to N.L.H.")
                    st.subheader("So it is not Recommended :thumbsdown:")
            else:
                pass
        else:
            pass
        button2 = st.checkbox('Random Forest Classifier')
        if button2:
            x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=50)

            model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
            model.fit(x_train, y_train)

            model.score(x_test, y_test)

            y_predicted_rfc = model.predict(x_test)

            c1 = confusion_matrix(y_test, y_predicted_rfc)

            accuracy = f1_score(y_test, y_predicted_rfc, average='weighted')

            y_predicted_rfc_custom = model.predict(X_scaled_custom)

            dot = export_graphviz(model.estimators_[499],
                                  filled=True,
                                  feature_names=('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'),
                                  class_names=('50', '80', '100', '200', '300', '400'),
                                  label="all")

            choose_dct1 = st.button('Model Accuracy')
            if choose_dct1:
                st.header('Model Accuracy')
                st.subheader(accuracy * 100)
            else:
                pass
            choose_dct2 = st.button('Heatmap')
            if choose_dct2:
                plt.figure(figsize=(8, 6))
                sns.heatmap(c1, annot=True)
                plt.xlabel('predict')
                plt.ylabel('Truth')
                st.pyplot(plt)
            else:
                pass
            choose_dct3 = st.button('Plot')
            if choose_dct3:
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(y_test.index, y_test, color='black')
                plt.xlabel('index')
                plt.ylabel('Test_data')
                plt.subplot(1, 2, 2)
                plt.scatter(y_test.index, y_predicted_rfc, color='green')
                plt.xlabel('index')
                plt.ylabel('Predicted_data')
                st.pyplot(plt)
            else:
                pass
            choose_dct4 = st.button('Visualize Model')
            if choose_dct4:
                st.graphviz_chart(dot)
            else:
                pass
            choose_dct5 = st.button('Custom Data Input Result')
            if choose_dct5:
                st.header("Custom output data")
                for i in range(0, len(y_predicted_rfc_custom)):
                    st.subheader(f"Input No. {i+1} the model predicted Concentration --> {y_predicted_rfc_custom[i]}")
            else:
                pass
            choose_dct6 = st.button('Suggestion on Custom Data')
            if choose_dct6:
                ans = mode(y_predicted_rfc_custom)
                sugg = ((ans / 1000000) / 180.16) * 1000000
                sugg1 = sugg * 1000
                st.subheader(f"In the Custom Input The Most Frequent Occurring Concentration is {ans} microMolar")
                st.subheader(f"Which is Equal to {sugg1} mg/L")
                if 225 < sugg1 < 1500:
                    st.subheader("Because the Current Concentration is in Ideal Range of 225 to 1500 mg/L according "
                                 "to N.L.H.")
                    st.subheader("So it is Recommended :thumbsup:")
                else:
                    st.subheader("Because the Current Concentration is not in Ideal Range of 225 to 1500 mg/L according"
                                 " to N.L.H.")
                    st.subheader("So it is not Recommended :thumbsdown:")
            else:
                pass
        else:
            pass
        button3 = st.checkbox('Xgboost Classifier')
        if button3:
            x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=50)

            le = LabelEncoder()
            le.fit(y_train)
            y_train = le.transform(y_train)
            model = XGBClassifier(max_depth=10000, n_estimators=10000, gamma=0.000000000000000001,
                                  min_child_weight=0.000000000000000001, reg_alpha=0.000000000000000001,
                                  reg_lambda=0.000000000000000001)
            model.fit(x_train, y_train)
            y_predicted_xgb = le.inverse_transform(model.predict(x_test))

            c1 = confusion_matrix(y_test, y_predicted_xgb)

            accuracy = f1_score(y_test, y_predicted_xgb, average='weighted')

            y_predicted_xgb_custom = model.predict(X_scaled_custom)
            y_predicted_xgb_custom_inv = le.inverse_transform(y_predicted_xgb_custom)

            choose_dct1 = st.button('Model Accuracy')
            if choose_dct1:
                st.header('Model Accuracy')
                st.subheader(accuracy * 100)
            else:
                pass
            choose_dct2 = st.button('Heatmap')
            if choose_dct2:
                plt.figure(figsize=(8, 6))
                sns.heatmap(c1, annot=True)
                plt.xlabel('predict')
                plt.ylabel('Truth')
                st.pyplot(plt)
            else:
                pass
            choose_dct3 = st.button('Plot')
            if choose_dct3:
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(y_test.index, y_test, color='black')
                plt.xlabel('index')
                plt.ylabel('Test_data')
                plt.subplot(1, 2, 2)
                plt.scatter(y_test.index, y_predicted_xgb, color='green')
                plt.xlabel('index')
                plt.ylabel('Predicted_data')
                st.pyplot(plt)
            else:
                pass
            choose_dct4 = st.button('Custom Data Input Result')
            if choose_dct4:
                st.header("Custom output data")
                for i in range(0, len(y_predicted_xgb_custom_inv)):
                    st.subheader(
                        f"Input No. {i+1} the model predicted Concentration --> {y_predicted_xgb_custom_inv[i]}")
            else:
                pass
            choose_dct5 = st.button('Suggestion on Custom Data')
            if choose_dct5:
                ans = mode(y_predicted_xgb_custom_inv)
                sugg = ((ans / 1000000) / 180.16) * 1000000
                sugg1 = sugg * 1000
                st.subheader(f"In the Custom Input The Most Frequent Occurring Concentration is {ans} microMolar")
                st.subheader(f"Which is Equal to {sugg1} mg/L")
                if 225 < sugg1 < 1500:
                    st.subheader("Because the Current Concentration is in Ideal Range of 225 to 1500 mg/L according "
                                 "to N.L.H.")
                    st.subheader("So it is Recommended :thumbsup:")
                else:
                    st.subheader("Because the Current Concentration is not in Ideal Range of 225 to 1500 mg/L according"
                                 " to N.L.H.")
                    st.subheader("So it is not Recommended :thumbsdown:")
            else:
                pass
        button4 = st.checkbox('Support Vector Machine')
        if button4:
            x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=50)
            classifier3 = SVC(kernel='rbf', random_state=6, cache_size=200, C=1000, gamma=1,
                              decision_function_shape='ovr', degree=3, coef0=0.00001, tol=0.0000001)
            classifier3.fit(x_train, y_train)
            y_predicted_svm = classifier3.predict(x_test)
            y_predicted_svm_custom = classifier3.predict(X_scaled_custom)
            accuracy = f1_score(y_test, y_predicted_svm, average='weighted')
            c1 = confusion_matrix(y_test, y_predicted_svm)

            choose_dct1 = st.button('Model Accuracy')
            if choose_dct1:
                st.header('Model Accuracy')
                st.subheader(accuracy * 100)
            else:
                pass
            choose_dct2 = st.button('Heatmap')
            if choose_dct2:
                plt.figure(figsize=(8, 6))
                sns.heatmap(c1, annot=True)
                plt.xlabel('predict')
                plt.ylabel('Truth')
                st.pyplot(plt)
            else:
                pass
            choose_dct3 = st.button('Plot')
            if choose_dct3:
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(y_test.index, y_test, color='black')
                plt.xlabel('index')
                plt.ylabel('Test_data')
                plt.subplot(1, 2, 2)
                plt.scatter(y_test.index, y_predicted_svm, color='green')
                plt.xlabel('index')
                plt.ylabel('Predicted_data')
                st.pyplot(plt)
            else:
                pass
            choose_dct4 = st.button('Custom Data Input Result')
            if choose_dct4:
                st.header("Custom output data")
                for i in range(0, len(y_predicted_svm_custom)):
                    st.subheader(
                        f"Input No. {i + 1} the model predicted Concentration --> {y_predicted_svm_custom[i]}")
            else:
                pass
            choose_dct5 = st.button('Suggestion on Custom Data')
            if choose_dct5:
                ans = mode(y_predicted_svm_custom)
                sugg = ((ans / 1000000) / 180.16) * 1000000
                sugg1 = sugg * 1000
                st.subheader(f"In the Custom Input The Most Frequent Occurring Concentration is {ans} microMolar")
                st.subheader(f"Which is Equal to {sugg1} mg/L")
                if 225 < sugg1 < 1500:
                    st.subheader("Because the Current Concentration is in Ideal Range of 225 to 1500 mg/L according "
                                 "to N.L.H.")
                    st.subheader("So it is Recommended :thumbsup:")
                else:
                    st.subheader("Because the Current Concentration is not in Ideal Range of 225 to 1500 mg/L according"
                                 " to N.L.H.")
                    st.subheader("So it is not Recommended :thumbsdown:")
            else:
                pass
        button4 = st.checkbox('K Nearest Neighbour')
        if button4:
            x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=50)
            classifier2 = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
            classifier2.fit(x_train, y_train)
            y_predicted_knn = classifier2.predict(x_test)
            y_predicted_knn_custom = classifier2.predict(X_scaled_custom)

            accuracy = f1_score(y_test, y_predicted_knn, average='weighted')
            c1 = confusion_matrix(y_test, y_predicted_knn)

            choose_dct1 = st.button('Model Accuracy')
            if choose_dct1:
                st.header('Model Accuracy')
                st.subheader(accuracy * 100)
            else:
                pass
            choose_dct2 = st.button('Heatmap')
            if choose_dct2:
                plt.figure(figsize=(8, 6))
                sns.heatmap(c1, annot=True)
                plt.xlabel('predict')
                plt.ylabel('Truth')
                st.pyplot(plt)
            else:
                pass
            choose_dct3 = st.button('Plot')
            if choose_dct3:
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(y_test.index, y_test, color='black')
                plt.xlabel('index')
                plt.ylabel('Test_data')
                plt.subplot(1, 2, 2)
                plt.scatter(y_test.index, y_predicted_knn, color='green')
                plt.xlabel('index')
                plt.ylabel('Predicted_data')
                st.pyplot(plt)
            else:
                pass
            choose_dct4 = st.button('Custom Data Input Result')
            if choose_dct4:
                st.header("Custom output data")
                for i in range(0, len(y_predicted_knn_custom)):
                    st.subheader(
                        f"Input No. {i + 1} the model predicted Concentration --> {y_predicted_knn_custom[i]}")
            else:
                pass
            choose_dct5 = st.button('Suggestion on Custom Data')
            if choose_dct5:
                ans = mode(y_predicted_knn_custom)
                sugg = ((ans / 1000000) / 180.16) * 1000000
                sugg1 = sugg * 1000
                st.subheader(f"In the Custom Input The Most Frequent Occurring Concentration is {ans} microMolar")
                st.subheader(f"Which is Equal to {sugg1} mg/L")
                if 225 < sugg1 < 1500:
                    st.subheader("Because the Current Concentration is in Ideal Range of 225 to 1500 mg/L according "
                                 "to N.L.H.")
                    st.subheader("So it is Recommended :thumbsup:")
                else:
                    st.subheader("Because the Current Concentration is not in Ideal Range of 225 to 1500 mg/L according"
                                 " to N.L.H.")
                    st.subheader("So it is not Recommended :thumbsdown:")
            else:
                pass
        else:
            pass

if s == "Devs":
    st.title(":man: Raktim Ghosal")
    st.header(':telephone_receiver: LinkedIn [link](https://www.linkedin.com/in/raktim-ghosal-069420dogg)')
    st.title(":man: Debdeep Banerjee")
    st.header(':telephone_receiver: LinkedIn [link](https://www.linkedin.com/in/debdeep-banerjee-8820161b3)')
    st.title(":man: Md Shakib")
    st.header(':telephone_receiver: LinkedIn [link](https://www.linkedin.com/in/sakibmd)')

if s == "Future Scope":
    st.header(":wrench: Hardware Integration")
    st.subheader(":clock1230: Coming Soon .............................")
    st.header(":wrench: Analysis on Multiple Chemical")
    st.subheader(":clock1230: Coming Soon .............................")

if s == "Help":
    st.header(":green[This is the Help Page, where a new user can get overview of the web application and can learn "
              "how to "
              "use it]")
    st.subheader("Step 1: Open Project Menu")
    st.image("App_project.png")
    st.subheader("Step 2: Upload the Data for Training")
    st.image("App_Data_Upload.png")
    st.image("App_After_Data_Upload.png")
    st.subheader("Step 3: Visualize the Dataframe")
    st.image("App_Dataframe_Visual.png")
    st.subheader("Step 4: Select a Model of Your Choice")
    st.image("App_Model_Select.png")
    st.subheader("Step 5: Get Insights on the Selected Model")
    st.image("App_Model_Accuracy.png")
    st.image("App_Model_Visualize.png")
    st.subheader("Step 6: Test on Custom Data Inputs")
    st.image("App_Custom_Output.png")
