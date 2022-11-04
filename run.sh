#!/bin/bash
echo "=> Waiting for Fakts Endpoint to be online"
arkitekt wait --retries=10 --interval=5 --silent

echo "=> Retrieving Fakts about local installation"
arkitekt init --grant=claim --silent

echo "=> Waiting for services to be healty"
arkitekt check --retries=3 --interval=5 --silent

echo "=> Starting app"
arkitekt run