#!/bin/bash
parts=(1 2 3 4 4b);
langs=(ES RU)

for j in ${parts[@]} ; do
  echo "=================================="
  echo "Running hmm_part_$j.py"
  python3 hmm_part_$j.py
  for lang in ${langs[@]} ; do
    echo "=================================="
    echo "Checking ${lang}/p$j.txt"
    python3 EvalScript/evalResult.py ${lang}/dev.out ${lang}/dev.p$j.out > ${lang}/p$j.txt
    cat ${lang}/p$j.txt
  done
done