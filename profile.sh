#!/bin/bash

python3 -m cprofilev -f constructor_profile_$(date +%Y-%m-%d_%H-%M-%S)

if [ -z ${BROWSER} ]; then
    firefox http://localhost:4000
else
    ${BROWSER} http://localhost:4000
fi
