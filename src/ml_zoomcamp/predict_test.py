#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import httpx

url = "http://localhost:9696/predict"

customer_id = "xyz-123"
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "tenure": 1,
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month_to_month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}


def main():
    response = httpx.post(url, json=customer).json()

    print(response)

    if response["churn"]:
        print(f"Sending promo email to {customer_id}")
    else:
        print(f"Not sending promo email to {customer_id}")


if __name__ == "__main__":
    main()
