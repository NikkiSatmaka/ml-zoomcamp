#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask

app = Flask("ping")


@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
