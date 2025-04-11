#!/bin/bash
cd ~/scripts
./create_nginx_conf.sh
./verify_services.sh
echo "Configuration updated. If you see services listening on ports 3000 and 3001, the 502 error should be resolved."
echo "Please try accessing https://astroshield.sdataplab.com again."
