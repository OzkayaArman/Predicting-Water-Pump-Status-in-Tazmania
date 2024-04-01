import pandas as pd
import seaborn as sns
import numpy as np
import warnings
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Set the option to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
#warnings.simplefilter(action='ignore', category=FutureWarning)

#Categorical Columns We Will Work With(Feature Selection Explained in the report)
categorical_cols = [
    'id','funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region',
    'lga', 'ward', 'public_meeting', 'permit', 'extraction_type_group',
    'management', 'payment_type', 'quality_group', 'quantity', 'source',
    'source_class', 'waterpoint_type'
]
# Numerical columns (Feature Selection Explained in the report)
numerical_cols = [
    'id', 'date_recorded', 'gps_height', 'longitude', 'latitude',
    'population', 'construction_year'
]

#GAP reference:https://www.youtube.com/watch?v=6eJHk
def scaleNumeric(df_numerical):
    scaleStandard = StandardScaler()
    columns_to_scale = df_numerical.columns.difference(['id'])
    scaled_data = scaleStandard.fit_transform(df_numerical[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=df_numerical.index)
    scaled_df = pd.concat([df_numerical[['id']], scaled_df], axis=1)
    return scaled_df

def scaleAllData(final_merged_df):
    scaleStandard = StandardScaler()
    columns_to_scale = final_merged_df.columns.difference(['status_group'])
    scaled_data = scaleStandard.fit_transform(final_merged_df[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=final_merged_df.index)
    final_merged_df = pd.concat([scaled_df,final_merged_df[['status_group']]], axis=1)
    return final_merged_df

#Encodes All Columns of Given DataFrame in OneHotEncoder
#GAP: https://www.youtube.com/watch?v=rsyrZnZ8J2o
#Returns given dataframe with ordinal encoding
def oneHotEncoderFunc(DataFrame):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    categorical_cols_without_id = [
        'funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region',
        'lga', 'ward', 'public_meeting', 'permit', 'extraction_type_group',
        'management', 'payment_type', 'quality_group', 'quantity', 'source',
        'source_class', 'waterpoint_type'
    ]

    ohetransform = ohe.fit_transform(DataFrame[categorical_cols_without_id])

    result_df = pd.concat([DataFrame, ohetransform], axis=1).drop(columns=categorical_cols)
    result_df = pd.concat([DataFrame['id'],result_df], axis=1)
    return result_df

# Encodes A column of Given DataFrame in ordinalEncoding. It Needs A For Loop to call Function for every column
# GAP reference: https://www.youtube.com/watch?v=15uClAVV-rI
def ordinalEncoderFunc(df_categorical):
    for column in df_categorical.columns:
        if column != 'id':
            orenc = OrdinalEncoder()
            encoded_data = orenc.fit_transform(df_categorical[[column]])
            df_categorical[column] = encoded_data
    return

#GAP reference: https://www.youtube.com/watch?v=nd7vc4MZQz4&t=277s
def targetEncodingFunc(df_categorical,labels):
    merge_cat_label_df= pd.merge(df_categorical, labels, on='id')
    target = 'status_group'
    cols = ['id','funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region',
        'lga', 'ward', 'public_meeting', 'permit', 'extraction_type_group',
        'management', 'payment_type', 'quality_group', 'quantity', 'source',
        'source_class', 'waterpoint_type']
    encoders = {}
    for column in cols:
        if column != 'id':
            te = TargetEncoder()
            te.fit(X=merge_cat_label_df[column], y=merge_cat_label_df[target])
            values = te.transform(merge_cat_label_df[column])
            merge_cat_label_df = pd.concat([merge_cat_label_df,values], axis=1)
            encoders[column] = te
    encoded_df = merge_cat_label_df.iloc[:, 20:39]
    final_df = pd.concat([merge_cat_label_df['id'], encoded_df], axis=1)
    return encoders, final_df

def targetEncodingFuncForTest(testdf, encoders):
    data_for_encoding = testdf.copy()
    for column, encoder in encoders.items():
        data_for_encoding[column] = encoder.transform(data_for_encoding[[column]])
    return data_for_encoding

def processUnknownObservationsSingleVar(DataFrame, ColName):
    # .mode()[0] - gives first category name
    most_frequent_category = DataFrame[ColName].mode()[0]
    #print("Most Frequent Category", most_frequent_category, '\n')

    # replace nan values with most occured category
    updated_df = DataFrame.copy()

    mask = updated_df[ColName].isna()
    updated_df.loc[mask, ColName] = most_frequent_category
    return updated_df

#GAP reference https://nbviewer.org/github/justmarkham/scikit-learn-tips/blob/master/notebooks/11_new_imputers.ipynb
def processUnknownObservationsWithMice(DataFrame):
    impute_it = IterativeImputer(verbose=2)
    imputed_array = impute_it.fit_transform(DataFrame)
    imputed_df = pd.DataFrame(imputed_array, columns=DataFrame.columns, index=DataFrame.index)
    return imputed_df

def preprocessing(train_input_file, train_labels_file,test_input_file):
    data = pd.read_csv(train_input_file)
    labels = pd.read_csv(train_labels_file)
    test = pd.read_csv(test_input_file)

    # Prepare labels for machine learning fitting
    labels['status_group'] = labels['status_group'].replace('functional needs repair', 'functional')

    #Replace unknown values with nan
    columns_except_id = [col for col in data.columns if col != 'id']
    for col in columns_except_id:
        data[col] = data[col].replace([0, 0.0, -0.00000002, 'unknown', 'none','other'], np.nan)

    columns_except_id = [col for col in test.columns if col != 'id']

    for col in columns_except_id:
        test[col] = test[col].replace([0, 0.0, -0.00000002, 'unknown', 'none', 'other'], np.nan)

    # Data Pre-Processing
    data = data.drop(
        ['amount_tsh', 'num_private', 'region_code', 'district_code', 'recorded_by', 'scheme_management', 'scheme_name',
         'extraction_type', 'extraction_type_class', 'management_group', 'payment', 'water_quality', 'quantity_group',
         'source_type', 'waterpoint_type_group'], axis=1)
    data = data.dropna(subset=['subvillage'])
    data = data.dropna(subset=['wpt_name'])
    data = data.dropna(subset=['longitude'])
    data = data.dropna(subset=['latitude'])

    test = test.drop(
        ['amount_tsh', 'num_private', 'region_code', 'district_code', 'recorded_by', 'scheme_management', 'scheme_name',
         'extraction_type', 'extraction_type_class', 'management_group', 'payment', 'water_quality', 'quantity_group',
         'source_type', 'waterpoint_type_group'], axis=1)


    # Verified Entire Dataset Observation Values with .unique()
    pump_status = ['non functional', 'functional']
    enc = OrdinalEncoder(categories=[pump_status])
    labels['status_group'] = enc.fit_transform(labels[['status_group']])

    #Split dataframe into 2 dataframe categorical and numerical
    df_categorical = data[categorical_cols]
    test_categorical = test[categorical_cols]

    df_numerical = data[numerical_cols]
    test_numerical = test[numerical_cols]


    # Turn date string into integer
    # Converting dates into separate year, month, and day features will allow the models to capture potential seasonal patterns over time.
    df_numerical['date_recorded'] = pd.to_datetime(df_numerical['date_recorded'])
    df_numerical.loc[:, 'year_recorded'] = df_numerical['date_recorded'].dt.year
    df_numerical.loc[:, 'month_recorded'] = df_numerical['date_recorded'].dt.month
    df_numerical.loc[:, 'day_recorded'] = df_numerical['date_recorded'].dt.day
    df_numerical = df_numerical.drop(['date_recorded'], axis=1)

    test_numerical['date_recorded'] = pd.to_datetime(test_numerical['date_recorded'])
    test_numerical.loc[:, 'year_recorded'] = test_numerical['date_recorded'].dt.year
    test_numerical.loc[:, 'month_recorded'] = test_numerical['date_recorded'].dt.month
    test_numerical.loc[:, 'day_recorded'] = test_numerical['date_recorded'].dt.day
    test_numerical = test_numerical.drop(['date_recorded'], axis=1)

    # Handle Missing Values of Numerical DataFrame
    df_numerical = processUnknownObservationsWithMice(df_numerical)
    test_numerical = processUnknownObservationsWithMice(test_numerical)

    # Handle Missing Values of Categorical DataFrame
    for column in df_categorical.columns:
        if column != 'id':
            df_categorical = processUnknownObservationsSingleVar(df_categorical, column)

    for column in test_categorical.columns:
        if column != 'id':
            test_categorical = processUnknownObservationsSingleVar(test_categorical, column)

    return df_categorical, df_numerical, labels, test_categorical, test_numerical, test
def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 8:
        for arguments in sys.argv:
            print(arguments)
        print(len(sys.argv))
        print(
            "Usage: python part1.py <train-input-file> <train-labels-file> <test-input-file> <numerical-preprocessing> <categorical-preprocessing> <model-type> <test-prediction-output-file>")
        sys.exit(1)  # Exit the script with an error code

    # Extract arguments
    train_input_file = sys.argv[1]
    train_labels_file = sys.argv[2]
    test_input_file = sys.argv[3]
    numerical_preprocessing = sys.argv[4]
    categorical_preprocessing = sys.argv[5]
    model_type = sys.argv[6]
    test_prediction_output_file = sys.argv[7]

    df_categorical, df_numerical, labels, test_categorical, test_numerical, test = preprocessing(train_input_file, train_labels_file, test_input_file)

    #print(df_numerical)
    #Numerical Preprocessing step
    if (numerical_preprocessing == "StandardScaler"):
        print("Running standard scaler")
        df_numerical = scaleNumeric(df_numerical)
        test_numerical = scaleNumeric(test_numerical)

    # Categorical Preprocessing step
    if (categorical_preprocessing == "OneHotEncoder"):
        print("Running OneHotEncoder")
        df_categorical = oneHotEncoderFunc(df_categorical)
        test_categorical = oneHotEncoderFunc(test_categorical)
        common_columns = df_categorical.columns.intersection(test_categorical.columns)
        df_categorical=df_categorical[common_columns]
        test_categorical=test_categorical[common_columns]

    if (categorical_preprocessing == "OrdinalEncoder"):
        print("Running Ordinal Encoder")
        ordinalEncoderFunc(df_categorical)
        ordinalEncoderFunc(test_categorical)

    if (categorical_preprocessing == "TargetEncoder"):
        print("Running Target Encoder")
        encoder, df_categorical = targetEncodingFunc(df_categorical, labels)
        test_categorical = targetEncodingFuncForTest(test_categorical, encoder)

    merged_df = pd.merge(df_categorical, df_numerical, on='id')
    merged_test = pd.merge(test_categorical, test_numerical, on='id')
    #Ensure column order between test and training df
    merged_test = merged_test[merged_df.columns]
    final_merged_df = pd.merge(merged_df, labels, on='id')
    final_merged_df = final_merged_df.drop(['id'], axis=1)
    merged_test = merged_test.drop(['id'], axis=1)
    #final_merged_df = scaleAllData(final_merged_df)
    X = final_merged_df.iloc[:, :-1]
    Y = final_merged_df.iloc[:, -1]

    if (model_type == "LogisticRegression"):
        print("Running LogisticRegression")
        model = LogisticRegression()
        scores = cross_val_score(estimator=model, X=X, y=Y, cv=5, scoring='accuracy')
        print("Cross-validation scores: ", scores)
        print("Mean cross-validation score: ", np.mean(scores))
        print("Standard deviation of cross-validation scores: ", np.std(scores))
        # Null Accuracy: The accuracy by predicting the most frequent class
        Y.value_counts(normalize=True)
        print(Y.value_counts(normalize=True))
        model.fit(X, Y)
        predictions = model.predict(merged_test)
        predictions_df = pd.DataFrame(predictions, columns=['status_group'])
        predictions_df['status_group'] = predictions_df['status_group'].map({1: 'functional', 0: 'non functional'})
        id_df = test[['id']]
        final_predictions_df = pd.concat([id_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
        final_predictions_df.to_csv(test_prediction_output_file, index=False)


    if (model_type == "RandomForestClassifier"):
        print("Running RandomForestClassifier")
        clf = RandomForestClassifier()
        scores = cross_val_score(estimator=clf, X=X, y=Y, cv=5, scoring='accuracy')
        print("Cross-validation scores: ", scores)
        print("Mean cross-validation score: ", np.mean(scores))
        print("Standard deviation of cross-validation scores: ", np.std(scores))
        print(Y.value_counts(normalize=True))
        clf.fit(X, Y)
        predictions = clf.predict(merged_test)
        predictions_df = pd.DataFrame(predictions, columns=['status_group'])
        predictions_df['status_group'] = predictions_df['status_group'].map({1: 'functional', 0: 'non functional'})
        id_df = test[['id']]
        final_predictions_df = pd.concat([id_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
        final_predictions_df.to_csv(test_prediction_output_file, index=False)

    if (model_type == "GradientBoostingClassifier"):
        print("Running GradientBoostingClassifier")
        gbr = GradientBoostingClassifier()
        scores = cross_val_score(estimator=gbr, X=X, y=Y, cv=5, scoring='accuracy')
        print("Cross-validation scores: ", scores)
        print("Mean cross-validation score: ", np.mean(scores))
        print("Standard deviation of cross-validation scores: ", np.std(scores))
        print(Y.value_counts(normalize=True))
        gbr.fit(X, Y)
        predictions = gbr.predict(merged_test)
        predictions_df = pd.DataFrame(predictions, columns=['status_group'])
        predictions_df['status_group'] = predictions_df['status_group'].map({1: 'functional', 0: 'non Functional'})
        id_df = test[['id']]
        final_predictions_df = pd.concat([id_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
        final_predictions_df.to_csv(test_prediction_output_file, index=False)

    if (model_type == "HistGradientBoostingClassifier"):
        print("Running HistGradientBoostingClassifier")
        histGbr = HistGradientBoostingClassifier()
        scores = cross_val_score(estimator=histGbr, X=X, y=Y, cv=5, scoring='accuracy')
        print("Cross-validation scores: ", scores)
        print("Mean cross-validation score: ", np.mean(scores))
        print("Standard deviation of cross-validation scores: ", np.std(scores))
        print(Y.value_counts(normalize=True))
        histGbr.fit(X, Y)
        predictions = histGbr.predict(merged_test)
        predictions_df = pd.DataFrame(predictions, columns=['status_group'])
        predictions_df['status_group'] = predictions_df['status_group'].map({1: 'functional', 0: 'non functional'})
        id_df = test[['id']]
        final_predictions_df = pd.concat([id_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
        final_predictions_df.to_csv(test_prediction_output_file, index=False)


    if (model_type == "MLPClassifier"):
        print("Running MLPClassifier")
        mlp = MLPClassifier(random_state=42, max_iter=300)
        scores = cross_val_score(estimator=mlp, X=X, y=Y, cv=5, scoring='accuracy')
        print("Cross-validation scores: ", scores)
        print("Mean cross-validation score: ", np.mean(scores))
        print("Standard deviation of cross-validation scores: ", np.std(scores))
        print(Y.value_counts(normalize=True))
        mlp.fit(X, Y)
        predictions = mlp.predict(merged_test)
        predictions_df = pd.DataFrame(predictions, columns=['status_group'])
        predictions_df['status_group'] = predictions_df['status_group'].map({1: 'functional', 0: 'non functional'})
        id_df = test[['id']]
        final_predictions_df = pd.concat([id_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
        final_predictions_df.to_csv(test_prediction_output_file, index=False)



warnings.filterwarnings("ignore")
# Calling Main Function
main()