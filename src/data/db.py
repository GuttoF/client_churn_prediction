import pickle
import duckdb

homepath = '/client_churn_prediction/src'
with open(homepath + 'data/processed/X_test.pkl', 'rb') as df:
    X_test = pickle.load(df)

with open(homepath + 'data/processed/y_test.pkl', 'rb') as df2:
    y_test = pickle.load(df2)

# Create a DuckDB database
con = duckdb.connect(database=(homepath + 'src/topbank.db'))

# Create a table
con.execute("CREATE TABLE X_test ("
            "age DOUBLE, "
            "credit_score_age_ratio DOUBLE, "
            "estimated_salary DOUBLE, "
            "balance DOUBLE, "
            "tenure DOUBLE, "
            "balance_salary_ratio DOUBLE, "
            "tenure_age_ratio DOUBLE, "
            "ltv DOUBLE, "
            "credit_score DOUBLE, "
            "num_of_products DOUBLE, "
            "balance_per_age DOUBLE, "
            "is_active_member_0 INT, "
            "is_active_member_1 INT, "
            "geography_france INT, "
            "geography_germany INT, "
            "geography_spain INT, "
            "gender_female INT, "
            "gender_male INT, "
            "life_stage_adolescence INT, "
            "life_stage_adulthood INT, "
            "life_stage_middle_age INT, "
            "life_stage_senior INT)")

X_test.to_sql('X_test', con, if_exists='replace', index=False)

# Create a table for the target variable
con.execute("CREATE TABLE y_test (exited INT)")

y_test.to_sql('y_test', con, if_exists='replace', index=False)

result = con.execute("SELECT * FROM X_test LIMIT 10")

print(result.fetchdf())