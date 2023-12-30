import pickle
import streamlit            as st
import pandas               as pd
import plotly.graph_objects as go
from plotly.subplots        import make_subplots
from typing                 import Union
from catboost               import CatBoostClassifier

# Title
st.title(':bank: Top Bank')
# Subheader
st.subheader('Client Churn Prediction')

st.markdown('''Considering a return per client:
- *15%* if the `estimated_salary` is lower than avg;
- *20%* if the `estimated_salary` is equal to the avg and is also less than two times the avg;
- *25%* if the `estimated_salary` is two times higher or more than avg;
- The return of all clients in this dataframe are: $38210856.42.''')

st.markdown('''The bank is losing $15689143.58, that value represents *41.05%* of the total return.''')

st.markdown('''The bank can deliver discount coupons to the top clients with highest probability of churn in simulation 1;''')
st.markdown('''The bank can select the optimal combination of clients that maximize the total returned value , without exceeding the total constraint in simulation 2.

Using the 0-1 knapsack-problem with probabilities with a budget of $10000.00:
- p(churn) >= 0.99: Client that will leave
- 0.95 <= p(churn) < 0.99: Client with a high probability to stay with a $200 coupon;
- 0.90 <= p(churn) < 0.99: Client that might stay with a $100 coupon;
- p(churn) < 0.90: Client that might stay with a $500 coupon.''')



def top_clients(scenario_name: str, data: Union[str, int, float], probability: str, prediction:str ,clients_return: str, churn_loss: float, number_of_clients: int, incentive_value: float):
    """
    Calculates the expected return and identifies the top clients to target for retention.

    Parameters:
    - scenario_name (str): Name of the scenario.
    - data (Union[str, int, float]): Additional data for the scenario.
    - probability (str): Probability column name in the dataframe.
    - clients_return (str): Column name for the clients' return in the dataframe.
    - churn_loss (float): Estimated loss due to churn per client.
    - number_of_clients (int): Number of top clients to target for retention.
    - incentive_value (float): Incentive value offered to the clients for retention.

    Returns:
    - top_clients_df (DataFrame): DataFrame containing the top clients to target for retention.
    - expected_return (float): Expected return in terms of revenue from the targeted clients.

    """
    # sort values by probability
    data.sort_values(by = probability, ascending = False, inplace = True)
    
    # select the top clients with the highest probability
    top_value = data.iloc[:number_of_clients, :]

    # send an incentive
    top_value['incentive'] = incentive_value

    # recover per client
    top_value['recover'] = top_value.apply(lambda x: x[clients_return] if x['exited'] == 1 else 0, axis = 1)

    # perfit
    top_value['profit'] = top_value['recover'] - top_value['incentive']

    # total recovered
    recovered_revenue = round(top_value['recover'].sum(), 2)

    # loss recovered in percentage
    loss_recovered = round(recovered_revenue / churn_loss * 100, 2)

    # sum of incentives
    sum_incentives = round(top_value['incentive'].sum(), 2)

    # profit sum
    profit_sum = round(top_value['profit'].sum(), 2)

    # ROI
    roi = round(profit_sum / sum_incentives * 100, 2)

    # calculate possible churn reduction in percentage
    churn_by_model = top_value[(top_value['exited'] == 1) & (top_value[prediction] == 1)]
    churn_real = round((len(churn_by_model) / len(data[data['exited'] == 1]))*100, 2)

    dataframe = pd.DataFrame({  'Scenario': scenario_name,
                                'Recovered Revenue': '$' + str(recovered_revenue),
                                'Loss Recovered': str(loss_recovered) + '%',
                                'Investment': '$' + str(sum_incentives),
                                'Profit': '$' + str(profit_sum),
                                'ROI': str(roi) + '%',
                                'Clients Recovered': str(len(churn_by_model)) + ' clients',
                                'Churn Reduction': str(churn_real) + '%'}, index = [0])

    return dataframe


def knapsack_solver(scenario_name: str, data: Union[str, int, float], prediction: str, clients_return: str, churn_loss: float, W: int, incentive: list):
    """
    A knapsack problem algorithm is a constructive approach to combinatorial optimization. Given set of items, each with a specific weight and a value. The algorithm determine each item's number to include with a total weight is less than a given limit.
    reference: https://www.geeksforgeeks.org/python-program-for-dynamic-programming-set-10-0-1-knapsack-problem/
    Parameters:
        W ([int]): [Capacity of the knapsack]
        wt ([list]): [Weight of the item]
        val ([list]): [Values of the item]
        name[str]: Name of the scenario
        data[DataFrame]: [The dataframe with the churn information]
        prediction_col[col string]: [Column of the dataframe with the model prediction]
        clients_return_col[col string]: [Column of the dataframe with the financial return of the clients]
        churn_loss[float]: [The loss of the banking with the churn]
        incentive_col[list]: [Discount value in a list]

    Returns:
        [DataFrame]: [A dataframe with the results calculated]
    """
    # filter clients in churn according model
    data = data[data[prediction] == 1]

    # set parameters for the knapsack function
    val = data[clients_return].astype(int).values # return per client
    wt = data[incentive].values # incentive value per client

    # number of itens in values
    n = len(val)

    # set K with 0 values
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
    max_val = K[n][W]

    # select items that maximizes the output
    keep = [False] * n
    res = max_val
    w = W
    for i in range(n, 0, -1):
        if res <= 0: break
        if res == K[i - 1][w]: continue
        else:
            keep[i - 1] = True
            res = res - val[i - 1]
            w = w - wt[i - 1]

    # dataframe with selected clients that maximizes output value
    data = data[keep]

    # Recover per client
    data['recover'] = data.apply(lambda x: x[clients_return] if x['exited'] == 1 else 0, axis = 1)
    
    # Calculate prefit
    data['profit'] = data['recover'] - data['incentive']
    
    # Calculate the total recovered
    recovered_revenue = round(data['recover'].sum(), 2)
    
    # Calculate loss recovered in percent
    loss_recovered = round((recovered_revenue/churn_loss)*100, 2)
    
    # Calculate the sum of incentives
    sum_incentives = round(data['incentive'].sum(), 2)
    
    # Calculate profit sum
    profit = round(data['profit'].sum(), 2)
    
    # Calculate ROI in percent
    roi = round((profit/sum_incentives)*100, 2)
    
    # Calculate possible churn reduction in %
    churn_by_model = data[(data['exited'] == 1) & (data[prediction] == 1)]
    churn_real = round((len(churn_by_model) / len(data[data['exited'] == 1]))*100, 2)
    
    dataframe = pd.DataFrame({ 'Scenario': scenario_name,
                            'Recovered Revenue': '$' + str(recovered_revenue),
                            'Loss Recovered': str(loss_recovered) + '%',
                            'Investment': '$' + str(sum_incentives),
                            'Profit': '$' + str(profit),
                            'ROI': str(roi) + '%',
                            'Clients Recovered': str(len(churn_by_model)) + ' clients',
                            'Churn Reduction': str(churn_real) + '%'}, index = [0])
    
    del K
    return dataframe

# Load data
X_data = pd.read_pickle('data/X_data.pkl')
y_data = pd.read_pickle('data/y_data.pkl')
df2 = pd.read_pickle('data/df2.pkl')
X_test = pd.read_pickle('data/X_test.pkl')
y_test = pd.read_pickle('data/y_test.pkl')

estimated_salary = pd.read_pickle('data/processed/estimated_salary.pkl')

# Load model and threshold
model = CatBoostClassifier()
model.load_model('models/model.cbm')
threshold = pickle.load(open('models/threshold.pkl', 'rb'))
yhat_proba = model.predict_proba(X_test)[: , 1]
yhat = (yhat_proba >= threshold).astype(int)

# Creating a dataframe with the results
# df2 is a dataframe before the transformations
salary_mean = round(df2['estimated_salary'].mean(), 2)

# Predictions and Results
y_test_frame = y_test.to_frame().reset_index(drop = True)
y_proba = pd.DataFrame(yhat_proba).rename(columns = {0: 'probability'}).reset_index(drop = True)
y_predict = pd.DataFrame(yhat).rename(columns = {0: 'prediction'}).reset_index(drop = True)

# Estimated salary without mms
estimated_salary_frame = estimated_salary.to_frame().reset_index(drop = True)

# Creating a dataframe with the results
df_simulation = pd.concat((y_test_frame, y_proba, y_predict, estimated_salary_frame), axis = 1)

# Verify threshold
df_simulation['threshold'] = df_simulation['probability'].apply(lambda x: 'negative' if x <= 0.4 else 'positive')

# Reorder columns
df_simulation = df_simulation[['estimated_salary' , 'exited', 'prediction', 'probability', 'threshold']]

#Considering a return per client:
#- *15%* if the `estimated_salary` is lower than avg;
#- *20%* if the `estimated_salary` is equal to the avg and is also less than two times the avg;
#- *25%* if the `estimated_salary` is two times higher or more than avg;

df_simulation['financial_return'] = df_simulation['estimated_salary'].apply(lambda x: x * 0.15 if x < salary_mean
                                                      else x * 0.2 if x >= salary_mean and x < 2 * salary_mean
                                                      else x * 0.25)

return_clients = round(df_simulation['financial_return'].sum(), 2)
print(f'The return of all clients in this dataframe are: ${return_clients}')

df_simulation['financial_return'] = df_simulation['estimated_salary'].apply(lambda x: x * 0.15 if x < salary_mean
                                                      else x * 0.2 if x >= salary_mean and x < 2 * salary_mean
                                                      else x * 0.25)

return_clients = round(df_simulation['financial_return'].sum(), 2)
print(f'The return of all clients in this dataframe are: ${return_clients}')

churn_loss = round(df_simulation[df_simulation['exited'] == 1]['financial_return'].sum(), 2)
print(f'The bank is losing ${churn_loss}, that value represents {round ((churn_loss/return_clients)*100, 2)}% of the total return.')

# Sidebar
credit_value = st.sidebar.slider('Credit Value', 0, 200, 25)
top_clients_value = st.sidebar.slider('Top Clients', 0, 100, 50)
invested_value_knapsack = st.sidebar.slider('Invested Value', 0, 10000, 100)

# Simulation
if st.button("Begin Simulation"):
    simulation_1 = top_clients('Simulation 1', df_simulation, 'probability','prediction', 'financial_return', churn_loss, top_clients_value, credit_value)

    df_simulation_2 = df_simulation[df_simulation['prediction'] == 1]
    incentives_list = [200, 100, 50]
    incentives = []
    n = len(df_simulation_2)
    # set incentive value according churn predicted probability
    for i in range(n):
        entry = df_simulation_2.iloc[i]
        if entry['probability'] >= 0.95 and entry['probability']:
            incentives.append(incentives_list[0])
        elif entry['probability'] >= 0.90 and entry['probability'] < 0.95:
            incentives.append(incentives_list[1])
        else:
            incentives.append(incentives_list[2])
    df_simulation_2['incentive'] = incentives
    simulation_2 = knapsack_solver('Simulation 2', df_simulation_2, 'prediction', 'financial_return', churn_loss, invested_value_knapsack, 'incentive')
    comparation = pd.concat([simulation_1, simulation_2], axis = 0, ignore_index = True)
    comparation
    
    # Layout para os gráficos usando make_subplots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Scatter Plot: Clients Recovered vs ROI/Churn Reduction',
                                        'Histogram of Churn Probabilities',
                                        'Bar Chart: Recovered Revenue vs Loss Recovered',
                                        'Line Chart: Profit vs Investment'],
                        horizontal_spacing=0.15, vertical_spacing=0.2)

    # Scatter plot for 'ROI'
    fig.add_trace(go.Scatter(x=comparation['Clients Recovered'], y=comparation['ROI'],
                             mode='lines', name='ROI vs Clients Recovered',
                             marker=dict(size=10, color='blue')), row=1, col=1)

    # Histogram for 'Churn Probabilities'
    fig.add_trace(go.Histogram(x=df_simulation['probability'], nbinsx=20, name='Churn Probabilities'), row=1, col=2)

    # Bar chart for 'Recovered Revenue' and 'Loss Recovered'
    fig.add_trace(go.Bar(x=comparation['Scenario'], y=comparation['Recovered Revenue'], name='Recovered Revenue'), row=2, col=1)
    fig.add_trace(go.Bar(x=comparation['Scenario'], y=comparation['Loss Recovered'], name='Loss Recovered'), row=2, col=1)

    # Line chart for 'Profit' and 'Investment'
    fig.add_trace(go.Scatter(x=comparation['Scenario'], y=comparation['Profit'], mode='lines', name='Profit'), row=2, col=2)
    fig.add_trace(go.Scatter(x=comparation['Scenario'], y=comparation['Investment'], mode='lines', name='Investment'), row=2, col=2)

    # Update layout
    fig.update_layout(title_text='Simulation Plots', showlegend=False)

    # Plotly chart
    st.plotly_chart(fig)
