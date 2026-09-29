#!/usr/bin/env bash
reachable_sites=()
for site in "https://ossci-datasets.s3.amazonaws.com/mnist" "http://yann.lecun.com/exdb/mnist"; do
    echo "Checking $site/t10k-labels-idx1-ubyte.gz"
    if curl -fsSI --max-time 12 "$site/t10k-labels-idx1-ubyte.gz" 2>/dev/null | grep -qE 'HTTP/.* 200'; then
        echo "Site $site is reachable"
        reachable_sites+=($site)
    else
        echo "Site $site is not reachable"
    fi
done

echo "Reachable sites: ${reachable_sites[@]}"

if [ ! -f t10k-labels.gz ]; then
    wget -O t10k-labels.gz "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
else
    echo "t10k-labels.gz already exists"
fi
