rm -f output.txt
python3 rate_bennet_trimer.py 2>> output.txt
rm -f ../MC_PYTHON_TEST/$1
cp output.txt ../MC_PYTHON_TEST/$1