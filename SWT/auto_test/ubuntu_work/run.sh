# compile driver generator
make
echo compilation done.
rm *.o

# run driver generator -> driver.c
./a.out -r $2 $3
echo gen driver done.
rm a.out

# compile test_code.c driver.c
gcc -o b.out $1 $3
echo driver compilation done.

# run driver
./b.out
rm b.out