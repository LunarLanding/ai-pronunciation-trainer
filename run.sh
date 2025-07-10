#!/usr/bin/env bash

flask --app webApp.py run --host=10.244.54.174 --port=5000 --cert=$HOME/.ssl/localhost.pem --key=$HOME/.ssl/localhost-key.pem #--debug