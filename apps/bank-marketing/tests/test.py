#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import httpx

url = "http://localhost:9696/predict"


client_ids = ["xyz-123", "abc-456"]
clients = [
    {
        "job": "management",
        "duration": 400,
        "poutcome": "success",
    },
    {
        "job": "student",
        "duration": 280,
        "poutcome": "failure",
    },
]


def predict(client_id, client):
    response = httpx.post(url, json=client).json()

    print(response)

    if response["subscribe"]:
        print(f"Calling {client_id} to offer term deposit")
    else:
        print(f"Not calling {client_id}")


def main():
    for client_id, client in zip(client_ids, clients):
        predict(client_id, client)


if __name__ == "__main__":
    main()
