#!/bin/bash

if [ -f ${1} ]; then
    python3 -m cprofilev -f constructor_profile_$(date +%Y-%m-%d_%H-%M-%S) ${1}

    if [ -z ${BROWSER} ]; then
        firefox http://localhost:4000
    else
        ${BROWSER} http://localhost:4000
    fi
else
    echo "file '${1}' does not exist!"
fi
