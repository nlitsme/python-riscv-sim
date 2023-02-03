# run all NF5 tests
# skipping all rv64, amo and float tests
python3 rvtst.py -l \
         | grep -v "/rv64" \
         | grep -v "/rv32[0-9a-z]*-.-f" \
         | grep -v "/rv32[0-9a-z]*-.-amo"  \
         | grep -v "/amo" \
         | grep -v "/f\|[/-]recoding\|[/-]move" \
         | grep -v "p-lrsc" \
         | grep -vw "ldst" \
         | while read f; do
    bash runtst.sh "$f"
    echo
done
