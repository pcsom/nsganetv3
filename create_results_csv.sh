#!/bin/bash
# Simple CSV creation without pandas
CORPUS=$1
OUTPUT=$2

echo "arch_id,ks,e,d,r,accuracy,status" > $OUTPUT

for dir in $CORPUS/arch_*/; do
    arch=$(basename $dir)
    arch_num=${arch#arch_}
    
    # Get config
    ks=$(cat $dir/config.json | grep '"ks":' | cut -d'[' -f2 | cut -d']' -f1 | tr -d ' ')
    e=$(cat $dir/config.json | grep '"e":' | cut -d'[' -f2 | cut -d']' -f1 | tr -d ' ')
    d=$(cat $dir/config.json | grep '"d":' | cut -d'[' -f2 | cut -d']' -f1 | tr -d ' ')
    r=$(cat $dir/config.json | grep '"r":' | awk '{print $2}')
    
    # Get accuracy and status
    if [ -f "$dir/train.err" ]; then
        acc=$(grep "Best metric:" $dir/train.err | tail -1 | awk '{print $4}')
        status="success"
    else
        acc=""
        status="failed"
    fi
    
    echo "$arch_num,\"$ks\",\"$e\",\"$d\",$r,$acc,$status" >> $OUTPUT
done

echo "Created $OUTPUT with $(tail -n +2 $OUTPUT | wc -l) architectures"
