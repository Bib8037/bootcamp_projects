
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
import shap
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import plotly.express as px

from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import altair as alt




#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#

st.title('🔥 App EDA : Exploratory Data Analysis')

st.subheader("""
# Hello Data Scientist and Data Analyst!!

This app will help you explore your query data
* show data statistic
* show data correlation
* clustering data
* prediction feature
***
""")

st.subheader('Input Data')


# Collects user input features into dataframe
st.sidebar.header('User Input Data')
uploaded_file = st.sidebar.file_uploader("1. Please Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)


    # if st.button('Generate Report'):
    #     profile=ProfileReport(df) 
    #     st_profile_report(profile)

  
    st.write("""
    ***
    """)
    #Slidebar - select X
    option_X = list(df.columns)
    option_X_allow = option_X[3:5]
    selected_option_X = st.sidebar.multiselect('3. Select X for model prediction',option_X,option_X_allow)


    #Slidebar - Specify parameter settings
    st.sidebar.header('Set Parameters')
    split_size = st.sidebar.slider('Data split ratio (% forTraining Set)',10,90,80,5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 70, 1)

    st.subheader('Output Data')
    st.sidebar.header('User Input Parameters')

    # selected_month = st.sidebar.selectbox('Month', list(reversed(range(1,8))))




    # def load_data(Month):
    #     df = pd.read_csv(uploaded_file)
    #     df['date'] = pd.to_datetime(df['timestamp'])
    #     df['Date'] = df['date'].apply(lambda date:date.date())
    #     df['Month'] = df['date'].apply(lambda date:date.month)
    #     df_select_month = df
    #     #df_select_month = df[df['Month'] == Month]
    #     return df_select_month


    # df = load_data(selected_month)





    # 0. show statistic 
    # 0.5 show clustering / anormaly
    # 1. show the feature important



    # 2. should change to manually selection important parameter



    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        modify = st.checkbox("Add filters")

        if not modify:
            return df

        df = df.copy()

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]

        return df
    df_filter = filter_dataframe(df)
    # column = df.columns
    # checked = []
    # for i in range(len(column)):
    #   agree = st.checkbox(column[i])
    #   if agree:
    #     checked.append(column[i])
    # st.write(list(checked))

    # for i in range(len(checked)):
    #   test = df[[checked[i]]]
    #   result = pd.concat(test)
    df_filter['PE'] =  df_filter['C3MC.AF']/df_filter['C2H4GROSSCAL1.CPV']
    df_CO2 = df_filter[['timestamp',
    'FI001.PV',
    'FI005.PV',
    # 'FICS004.PV',
    # 'FICS009.PV',
    'S001NPARAFFINS-MOC.LAB',
    # 'S001ISOPARAFFINS-MOC.LAB',
    # 'S001NAPHTHENES-MOC.LAB',
    # 'S001AROMATICS-MOC.LAB',
    # 'S001SPGR60/60F-MOC.LAB',
    'AI001C.PV',
    'C2H4GROSSCAL1.CPV',
    'C3MC.AF',
    'PE',
    'FG',
    'TOTAL_CO2',
    'level CO2']]



    st.write('Data Dimension: ' + str( df_CO2.shape[0]) + ' rows and ' + str( df_CO2.shape[1]) + ' columns.')
    st.dataframe(df_CO2)
    st.write('---')




    #Graph chart
    df_filter['Date'] = pd.to_datetime(df_filter['timestamp'])
    st.subheader('CO2 emission from other source')
    st.line_chart(
        df_filter,
        x="Date",
        y=["EE","CKB","FLARE","STEAM"])  # <-- You can pass multiple columns!


    #Box chart
    fig = px.box(df_filter, x = ["EE","CKB","FLARE","STEAM",],
        hover_data = ["TOTAL_CO2"],
        title = "CO2 Emission Excluding FG",
        width = 700 )
    st.write(fig)

    #Graph chart
    df_filter['Date'] = pd.to_datetime(df_filter['timestamp'])
    st.subheader('CO2 emission from FG')
    st.line_chart(
        df_filter,
        x="Date",
        y=["FG"])  # <-- You can pass multiple columns!

    #Box chart
    fig = px.box(df_filter, x = ["FG"],
        hover_data = ["TOTAL_CO2"],
        title = "CO2 Emission From FG",
        width = 700 )
    st.write(fig)

    st.write('---')

    df1 = df_CO2.drop(['timestamp'], axis=1)

    #HeatMap correlation
    st.subheader('Intercorrelatiion Matrix Heatmap')
    corr = df1.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)
    st.write('---')

    #Grouping Data by catergoly
    df2 = df1.groupby('level CO2').median()
    st.subheader('Clustering : Median of each parameter')
    st.write(df2)
    st.bar_chart(df2)
    st.write('---')

    #Plotly.express

    # st.subheader('Clustering : Scatter Matrix 3D')
    # fig_3D = px.scatter_3d(df_filter, x= "FI001.PV", y='PE',z='AI001C.PV', color ='level CO2', opacity=0.5)
    # st.write(fig_3D)
    # st.write('---')

    st.subheader('Clustering : Scatter Matrix 2D')
    fig_2D = px.scatter_matrix(df_filter, 
    dimensions=["FI001.PV", "FI005.PV",'S001NPARAFFINS-MOC.LAB', "AI001C.PV", 'C2H4GROSSCAL1.CPV','C3MC.AF','PE'], 
    color="level CO2",
    width = 1200,
    height = 900 )
    st.write(fig_2D)
    st.write('---')





    # df = pd.DataFrame(data[symbol].Close)
    df_filter['Date'] = pd.to_datetime(df_filter['timestamp'])
    st.subheader('CO2 emission from FG VS Total')

    st.line_chart(
        df_filter,
        x="Date",
        y=['TOTAL_CO2',"FG"]) #"EE","CKB","FLARE","STEAM",
    st.write('---')



    def user_input_features():
        Naphtha_Flow = st.sidebar.slider('Naphtha_Flow', float(df_CO2[['FI001.PV']].min()), float(df_CO2[['FI001.PV']].max()), float(df_CO2[['FI001.PV']].mean()))
        LPG_Flow = st.sidebar.slider('LPG_Flow', float(df_CO2[['FI005.PV']].min()), float(df_CO2[['FI005.PV']].max()), float(df_CO2[['FI005.PV']].mean()))
        nPa = st.sidebar.slider('nPa', float(df_CO2[['S001NPARAFFINS-MOC.LAB']].min()), float(df_CO2[['S001NPARAFFINS-MOC.LAB']].max()), float(df_CO2[['S001NPARAFFINS-MOC.LAB']].mean()))
        Density = st.sidebar.slider('Density', float(df_CO2[['AI001C.PV']].min()), float(df_CO2[['AI001C.PV']].max()), float(df_CO2[['AI001C.PV']].mean()))
        C2_Product = st.sidebar.slider('C2_Product', float(df_CO2[['C2H4GROSSCAL1.CPV']].min()), float(df_CO2[['C2H4GROSSCAL1.CPV']].max()), float(df_CO2[['C2H4GROSSCAL1.CPV']].mean()))
        C3_Product = st.sidebar.slider('C3_Product', float(df_CO2[['C3MC.AF']].min()), float(df_CO2[['C3MC.AF']].max()), float(df_CO2[['C3MC.AF']].mean()))
        data = {'FI001.PV': Naphtha_Flow,
                'FI005.PV': LPG_Flow,
                'S001NPARAFFINS-MOC.LAB' : nPa,
                'AI001C.PV': Density,
                'C2H4GROSSCAL1.CPV': C2_Product,
                'C3MC.AF': C3_Product}
        features = pd.DataFrame(data, index=[0])
        return features

    df_feature = user_input_features()

    st.subheader('Prediction Model')

    df_model = df_CO2.dropna()
    X = df_model[['FI001.PV','FI005.PV','S001NPARAFFINS-MOC.LAB','AI001C.PV','C2H4GROSSCAL1.CPV','C3MC.AF']] #selected_option_X
    y = df_model['TOTAL_CO2']

    # st.markdown('**Dataset dimension**')
    # st.write('X')
    # st.info(X.shape)
    # st.write('Y')
    # st.info(y.shape)

    st.markdown('**Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(y.name)
    st.write('---')

    st.subheader('User Input Parameters for Prediction')
    st.write(df_feature)
    st.write('---')

    #slite train and test
    test_size= 1- (split_size/100)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state= seed_number)

    #Check accuracy of model
    algo = [[RandomForestRegressor(), 'RandomForestRegressor'],
    [SVR(), 'SVR'],
    [NuSVR(), 'NuSVR'],
    [BaggingRegressor(), 'BaggingRegressor'],
    [AdaBoostRegressor(), 'AdaBoostRegressor'],
    [ExtraTreesRegressor(), 'ExtraTreesRegressor'],
    [LinearRegression(), 'LinearRegression'],
    [GradientBoostingRegressor(), 'GradientBoostingRegressor']]

    for a in algo:
        model = a[0]
        model.fit(X_train,y_train)
        prediction=model.predict(df_feature)
        st.write('**Prediction from model**'+" : "+str(a[1])+ " = "+ str(round(prediction[0],2))+" ton CO2/h " )


        #Train
        y_prediction_train = model.predict(X_train)
        # st.write('Train : Mean squared error (MSE): %.2f'
        #   % mse(y_prediction_train, y_train))
        st.write('Train : Mean absolute error (MAE): %.2f'
        % mae(y_prediction_train, y_train))
        st.write('Train : Coefficient of determination (R^2): %.2f'
        % r2_score(y_prediction_train, y_train))

        #Test
        y_prediction_test = model.predict(X_test)
        # st.write('Test : Mean squared error (MSE): %.2f'
        #   % mse(y_prediction_test, y_test))
        st.write('Test : Mean absolute error (MAE): %.2f'
        % mae(y_prediction_test, y_test))
        st.write('Test : Coefficient of determination (R^2): %.2f'
        % r2_score(y_prediction_test, y_test))
        if r2_score(y_prediction_train, y_train)<0.85 or r2_score(y_prediction_test, y_test) <0.85 or mae(y_prediction_train, y_train) > 4 or mae(y_prediction_test, y_test)>4:
            st.write('**❌ Model Underfit**')
        elif ((mae(y_prediction_train, y_train) - mae(y_prediction_test, y_test)) <=2) and ((mae(y_prediction_test, y_test) - mae(y_prediction_train, y_train)) <=2): 
            st.write('**✅ Good Model**')
        else:
            st.write('**❌ Model Overderfit**')

        #Plotly.express
        fig = px.scatter(df, x = y_prediction_test, y= y_test, labels=dict(x="y_prediction_test", y="y_test"))
        st.write(fig)
        st.write('---')

    # Explaining the model's predictions using SHAP values
    # https://github.com/slundberg/shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots()
    st.subheader('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    # st.pyplot(bbox_inches='tight')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    # st.pyplot(bbox_inches='tight')
    st.pyplot(fig)



else:
    st.write('No data source')
   


#3. input for prediction
#4. load model
#5. prediction



