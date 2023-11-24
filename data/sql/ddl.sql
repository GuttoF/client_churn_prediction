-- Description: DDL for the churn_rate schema
-- Author: Gutto França
-- Last Modified: 2023-11-24
-- Last Modified By: Gutto França
-- License: MIT
-- Path: data/sql/ddl.sql

-- Create the customers table
CREATE TABLE IF NOT EXISTS churn_rate.customers (
    RowNumber INT,
    CustomerId INT PRIMARY KEY,
    Surname VARCHAR(255),
    CreditScore INT,
    Geography VARCHAR(255),
    Gender VARCHAR(10),
    Age INT,
    Tenure INT,
    Balance DECIMAL(20,2),
    NumOfProducts INT,
    HasCrCard BOOLEAN,
    IsActiveMember BOOLEAN,
    EstimatedSalary DECIMAL(20,2),
    Exited BOOLEAN
);

