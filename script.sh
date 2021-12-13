#!/bin/bash
parts=(1 2 3 4 4b);
langs=(ES RU)
test=(4 4b);

for i in "${parts[@]}" ; do
  echo "=================================="
  echo "Running hmm_part_$i.py for dev.in"
  python3 hmm_part_$i.py -t train -i dev.in
  for lang in "${langs[@]}" ; do
    echo "=================================="
    echo "Checking ${lang}/p$i.txt"
    python3 EvalScript/evalResult.py ${lang}/dev.out ${lang}/dev.p$i.out > ${lang}/p$i.txt
    cat ${lang}/p$i.txt
  done
done

for j in "${test[@]}" ; do
  echo "=================================="
  echo "Running hmm_part_$j.py for test.in"
  python3 hmm_part_$j.py -t train -i test.in
  for lang in "${langs[@]}" ; do
    echo "=================================="
    echo "Checking ${lang}/t$j.txt"
    python3 EvalScript/evalResult.py ${lang}/dev.out ${lang}/test.p$j.out > ${lang}/t$j.txt
    cat ${lang}/t$j.txt
  done
done