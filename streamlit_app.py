import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
st.title("Openlearn Capstone Project")
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load dataset once
df = pd.read_csv('survey.csv')

# Sidebar navigation with only EDA
section = st.sidebar.radio(
    "Select Section",
    [
        "EDA",
        "Regression",
        "Classification",
        "Clustering"
    ]
)

if section == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    # Nested selection inside EDA for the two charts
    eda_section = st.selectbox(
        "Choose EDA Chart",
        [
            "Gender Treatment Distribution",
            "Self-Employed Work Interference"
        ]
    )

    if eda_section == "Gender Treatment Distribution":
        st.subheader("Gender Distribution Among Those Who Sought Mental Health Treatment")

        # Gender mapping and filtering code
        gender_map = {
            'Female': 'F', 'female': 'F', 'F': 'F', 'f': 'F', 'Woman': 'F', 'woman': 'F',
            'Cis Female': 'F', 'cis-female/femme': 'F', 'Female (cis)': 'F', 'femail': 'F',
            'Trans-female': 'F', 'Trans woman': 'F', 'Female (trans)': 'F',
            'Femake': 'F', 'Female ': 'F',
            'Male': 'M', 'male': 'M', 'M': 'M', 'm': 'M', 'Man': 'M',
            'Cis Male': 'M', 'cis male': 'M', 'Male (CIS)': 'M', 'Cis Man': 'M',
            'maile': 'M', 'Mal': 'M', 'Make': 'M', 'msle': 'M', 'Mail': 'M',
            'Malr': 'M', 'Male ': 'M',
            'Male-ish': 'M', 'something kinda male?': 'M', 'Guy (-ish) ^_^': 'M',
            'male leaning androgynous': 'M',
            'ostensibly male, unsure what that really means': 'M'
        }

        data = df.copy()
        data = data[data['mental_health_consequence'] == "Yes"]
        data['Gender'] = data['Gender'].map(gender_map)
        data = data[data['Gender'].isin(['M', 'F'])]
        data = data[data['treatment'] == 'Yes']

        gender_counts = data['Gender'].value_counts()
        gender_df = gender_counts.rename_axis('Gender').reset_index(name='Count')
        gender_df['Gender'] = gender_df['Gender'].map({'M': 'Male', 'F': 'Female'})

        if gender_df.empty:
            st.warning("No data to display after filtering.")
        else:
            fig = px.pie(
                gender_df,
                names='Gender',
                values='Count',
                color='Gender',
                color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'},
                title='Percentage of Males vs Females Who Sought Help'
            )
            fig.update_traces(textinfo='percent+label', pull=[0.05, 0])
            fig.update_layout(showlegend=True, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

    elif eda_section == "Self-Employed Work Interference":
        st.subheader("Work Interference Among Self-Employed Individuals")

        interference_map = {
            'Often': 5,
            'Sometimes': 3,
            'Rarely': 1,
            'Never': 0,
            'No data': -1
        }
        df['work_interfere'] = df['work_interfere'].map(interference_map)

        interesting = df[df['self_employed'] == "Yes"]

        counts = interesting['work_interfere'].value_counts()
        labels_map = {
            5: 'Often',
            3: 'Sometimes',
            1: 'Rarely',
            0: 'Never',
            -1: 'No data'
        }
        plot_df = pd.DataFrame({
            'Level': counts.index.map(labels_map),
            'Count': counts.values
        })

        if plot_df.empty:
            st.warning("No self-employed data found for work interference.")
        else:
            fig = px.pie(
                plot_df,
                names='Level',
                values='Count',
                title="Self-Employed Individuals And Work Interference with Mental Health",
                color='Level',
                color_discrete_map={
                    'Often': '#e74c3c',
                    'Sometimes': '#f39c12',
                    'Rarely': '#27ae60',
                    'Never': '#3498db',
                    'No data': '#95a5a6'
                }
            )
            fig.update_traces(textinfo='percent+label', pull=[0.05]*len(plot_df))
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            📌 This pie chart shows how **self-employed** individuals report the frequency with which their work interferes with their **mental health**.
            """)


# Preprocessing
df = pd.read_csv('survey.csv')
import matplotlib.pyplot as plt
df['comments'] = df['comments'].fillna("")
df['self_employed'] = df['self_employed'].fillna(0)
df['state'] = df['state'].fillna("Unknown")
df['work_interfere'] = df['work_interfere'].fillna("No data")
replacement_map = {'Yes': 1, 'No': -1,0:0}
df['self_employed'] = df['self_employed'].map(replacement_map)
df['work_interfere'].unique()
m = {
    'Often':5,
    'Rarely':1,
    'Never':0,
    'Sometimes':3,
    'No data':-1
}
df['work_interfere'] = df['work_interfere'].map(m)
df['leave'].unique()
leave_map = {
    'Very easy': 4,
    'Somewhat easy': 3,
    'Somewhat difficult': 2,
    'Very difficult': 1,
    "Don't know": 0
}

df['leave'] = df['leave'].map(leave_map)
df['family_history'].unique()
map = {'Yes':1,'No':0}
df['family_history'] = df['family_history'].map(map)

df.loc[(df['Age'] < 18) | (df['Age'] > 75), 'Age'] = np.nan
df = df.dropna()
df['mental_health_consequence'].unique()
map = {
    'No':-1,
    'Yes':1,
    'Maybe':0
}
df['mental_health_consequence'] = df['mental_health_consequence'].map(map)

df['phys_health_consequence'].unique()
map = {
    'No':-1,
    'Yes':1,
    'Maybe':0
}
df['phys_health_consequence'] = df['phys_health_consequence'].map(map)

df['supervisor'].unique()
map = {
    'No':-1,
    'Yes':1,
    'Some of them':0
}
df['supervisor'] = df['supervisor'].map(map)

gender_map = {
    # Female variations
    'Female': 'F', 'female': 'F', 'F': 'F', 'f': 'F', 'Woman': 'F', 'woman': 'F',
    'Cis Female': 'F', 'cis-female/femme': 'F', 'Female (cis)': 'F', 'femail': 'F',
    'Trans-female': 'F', 'Trans woman': 'F', 'Female (trans)': 'F',
    'Femake': 'F', 'Female ': 'F',

    # Male variations
    'Male': 'M', 'male': 'M', 'M': 'M', 'm': 'M', 'Man': 'M',
    'Cis Male': 'M', 'cis male': 'M', 'Male (CIS)': 'M', 'Cis Man': 'M',
    'maile': 'M', 'Mal': 'M', 'Make': 'M', 'msle': 'M', 'Mail': 'M',
    'Malr': 'M', 'Male ': 'M',

    # Ambiguous but male-leaning
    'Male-ish': 'M', 'something kinda male?': 'M', 'Guy (-ish) ^_^': 'M',
    'male leaning androgynous': 'M',
    'ostensibly male, unsure what that really means': 'M'
}

df['Gender'] = df['Gender'].map(gender_map)


df['mental_health_interview'].unique()

map = {
    'No':-1,
    'Yes':1,
    'Maybe':0
}

df['mental_health_interview'] = df['mental_health_interview'].map(map)

df['phys_health_interview'].unique()

map = {
    'No':-1,
    'Yes':1,
    'Maybe':0
}

df['phys_health_interview'] = df['phys_health_interview'].map(map)

df['treatment'].unique()

map = {
    'No':0,
    'Yes':1,
  
}

df['treatment'] = df['treatment'].map(map)

map = {
    'No':0,
    'Yes':1,
    "Don't know":-1
}

df['mental_vs_physical'] = df['mental_vs_physical'].map(map)

map = {
    'No':0,
    'Yes':1,
  
}

df['obs_consequence'] = df['obs_consequence'].map(map)

df['coworkers'].unique()

map = {
    'No':-1,
    'Some of them':0,
    'Yes':1
}

df['coworkers'] = df['coworkers'].map(map)

df['no_employees'].unique()

map = {
    '6-25':1,
    'More than 1000':5,
    '26-100':2,
    '100-500':3,
    '1-5':0,
    '500-1000':4
}

df['no_employees'] = df['no_employees'].map(map)
map = {
    'Yes':1,
    'No':0
}
df['remote_work'] = df['remote_work'].map(map)


map = {
    'Yes':1,
    'No':0
}

df['tech_company'] = df['tech_company'].map(map)

map = {
    'Yes':1,
    'No':0,
    "Don't know":-1,
}

df['benefits'] = df['benefits'].map(map)
df = df.drop(['Country'],axis=1)

df = df.drop(['Timestamp'],axis=1)
df = df.drop(['state'],axis=1)

df['Gender'] = df['Gender'].fillna("O")

map = {
    'M':1,
    'F':0,
    'O':-1
}

df['Gender'] = df['Gender'].map(map)



map = {
    'Yes':1,
    'No':0,
    'Not sure':-1
}

df['care_options'] = df['care_options'].map(map)

map = {
    'Yes':1,
    'No':0,
    "Don't know":-1,
}

df['wellness_program'] = df['wellness_program'].map(map)
df['seek_help'] = df['seek_help'].map(map)
df['anonymity'] = df['anonymity'].map(map)
df = df.drop(['comments'],axis=1)

if section == "Regression":
        
    features = ['Gender', 'mental_health_interview', 'phys_health_interview', 'treatment',
                'mental_vs_physical', 'obs_consequence', 'coworkers']
    
    X = df.drop(['Age'],axis=1)
    y = df['Age']
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    lr_mae, lr_rmse, lr_r2 = calc_metrics(y_test, lr_preds)
    rf_mae, rf_rmse, rf_r2 = calc_metrics(y_test, rf_preds)

    st.header("Regression Models: Predict Age")

    model_choice = st.selectbox("Choose Regression Model", ["Linear Regression", "Random Forest"])

    st.subheader("Model Performance on Test Data:")

    if model_choice == "Linear Regression":
        st.write(f"MAE: {lr_mae:.2f}")
        st.write(f"RMSE: {lr_rmse:.2f}")
        st.write(f"R² Score: {lr_r2:.2f}")
    else:
        st.write(f"MAE: {rf_mae:.2f}")
        st.write(f"RMSE: {rf_rmse:.2f}")
        st.write(f"R² Score: {rf_r2:.2f}")

    st.markdown("---")

    st.subheader("Predict Age for Custom Input")

    maps = {
        'Gender': {'M': 1, 'F': 0},
        'mental_health_interview': {'No': -1, 'Maybe': 0, 'Yes': 1},
        'phys_health_interview': {'No': -1, 'Maybe': 0, 'Yes': 1},
        'treatment': {'No': 0, 'Yes': 1},
        'mental_vs_physical': {'No': 0, "Don't know": -1, 'Yes': 1},
        'obs_consequence': {'No': 0, 'Yes': 1},
        'coworkers': {'No': -1, 'Some of them': 0, 'Yes': 1},
    }

    all_features = X.columns.tolist()


    gender_input = st.selectbox("Gender", options=['M', 'F'], index=0)
    treatment_input = st.selectbox("Received Treatment", options=['No', 'Yes'], index=1)
    mental_health_interview_input = st.selectbox("Mental Health Interview", options=['No', 'Maybe', 'Yes'], index=2)
    phys_health_interview_input = st.selectbox("Physical Health Interview", options=['No', 'Maybe', 'Yes'], index=2)
    mental_vs_physical_input = st.selectbox("Mental vs Physical", options=['No', "Don't know", 'Yes'], index=0)
    obs_consequence_input = st.selectbox("Observed Consequence", options=['No', 'Yes'], index=0)
    coworkers_input = st.selectbox("Coworkers Support", options=['No', 'Some of them', 'Yes'], index=2)

    user_inputs = {
        'Gender': gender_input,
        'mental_health_interview': mental_health_interview_input,
        'phys_health_interview': phys_health_interview_input,
        'treatment': treatment_input,
        'mental_vs_physical': mental_vs_physical_input,
        'obs_consequence': obs_consequence_input,
        'coworkers': coworkers_input,
    }

    input_dict = {}
    for feature in all_features:
        if feature in user_inputs:
            val = user_inputs[feature]
            if feature in maps:
                input_dict[feature] = maps[feature].get(val, df[feature].mode()[0])  
            else:
                input_dict[feature] = val
        else:
            if pd.api.types.is_numeric_dtype(df[feature]):
                input_dict[feature] = df[feature].mean()
            else:
                input_dict[feature] = df[feature].mode()[0]

    input_df = pd.DataFrame([input_dict])

    if model_choice == "Linear Regression":
        predicted_age = lr_model.predict(input_df)[0]
    else:
        predicted_age = rf_model.predict(input_df)[0]

    st.write(f"### Predicted Age: {predicted_age:.1f} years")

if section == "Classification":
    st.header("Classification Models: Predict Treatment")

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

    y = df['treatment']
    X = df.drop(['treatment'], axis=1)

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.4, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Support Vector Machine": SVC(probability=True),
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)

    model_choice = st.selectbox("Choose Classification Model", list(models.keys()))
    model = models[model_choice]

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    st.subheader(f"Model Performance on Test Data: {model_choice}")
    st.text(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f"ROC-AUC Score: {auc:.3f}")
    con = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(con)

    st.markdown("---")

    st.subheader("Predict Treatment for Custom Input")

 
    gender_input = st.selectbox("Gender", ['M', 'F'])
    mental_health_interview_input = st.selectbox("Mental Health Interview", ['No', 'Maybe', 'Yes'])
    phys_health_interview_input = st.selectbox("Physical Health Interview", ['No', 'Maybe', 'Yes'])
    mental_vs_physical_input = st.selectbox("Mental vs Physical", ['No', "Don't know", 'Yes'])
    coworkers_input = st.selectbox("Coworkers Support", ['No', 'Some of them', 'Yes'])

    maps = {
        'Gender': {'M': 1, 'F': 0},
        'mental_health_interview': {'No': -1, 'Maybe': 0, 'Yes': 1},
        'phys_health_interview': {'No': -1, 'Maybe': 0, 'Yes': 1},
        'mental_vs_physical': {'No': 0, "Don't know": -1, 'Yes': 1},
        'coworkers': {'No': -1, 'Some of them': 0, 'Yes': 1},
    }

    input_dict = {}

    for feature in X.columns:
        if feature == 'Gender':
            input_dict[feature] = maps['Gender'][gender_input]
        elif feature == 'mental_health_interview':
            input_dict[feature] = maps['mental_health_interview'][mental_health_interview_input]
        elif feature == 'phys_health_interview':
            input_dict[feature] = maps['phys_health_interview'][phys_health_interview_input]
        elif feature == 'mental_vs_physical':
            input_dict[feature] = maps['mental_vs_physical'][mental_vs_physical_input]
        elif feature == 'coworkers':
            input_dict[feature] = maps['coworkers'][coworkers_input]
        else:
            if df[feature].dtype in [np.float64, np.int64]:
                input_dict[feature] = df[feature].mean()
            else:
                input_dict[feature] = df[feature].mode()[0]

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    for col in X_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_encoded.columns]  # reorder to match

    user_pred = model.predict(input_encoded)[0]
    user_pred_proba = model.predict_proba(input_encoded)[0, 1]

    st.write(f"### Predicted Treatment: {'Yes' if user_pred == 1 else 'No'}")
    st.write(f"### Prediction Probability (Yes): {user_pred_proba:.2f}")

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


if section == "Clustering":
    st.header("KMeans Clustering with User-Selected Features")

    available_features = [
        'family_history',
        'treatment',
        'work_interfere',
        'benefits',
        'care_options',
        'wellness_program',
        'seek_help',
        'anonymity',
        'leave',
        'mental_health_consequence',
        'mental_health_interview',
        'mental_vs_physical',
        'obs_consequence'
    ]

    selected_features = st.multiselect(
        "Select features for clustering (at least 2)",
        options=available_features,
        default=['family_history', 'mental_health_consequence', 'mental_health_interview']
    )

    if len(selected_features) < 2:
        st.warning("Please select at least two features for clustering.")
    else:
        X = df[selected_features]

        encoder = OneHotEncoder(sparse_output=False)

        X_encoded = encoder.fit_transform(X)

        n_clusters = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=2)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_encoded)

        df['cluster'] = kmeans.labels_

        score = silhouette_score(X_encoded, kmeans.labels_)
        st.write(f"Silhouette Score: **{score:.4f}**")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_encoded)

   
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=40)
        ax.set_title('Clusters Visualized by PCA Components 1 & 2')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        st.pyplot(fig)
