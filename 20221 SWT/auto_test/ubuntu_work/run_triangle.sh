# compile driver generator
make
echo compilation done.
rm *.o

# run driver generator -> driver.c
./a.out -r triangle_test_cases.txt triangle_driver.c
echo gen driver done.
rm a.out

# compile test_code.c driver.c
gcc -o b.out triangle.c triangle_driver.c
echo driver compilation done.

# run driver
./b.out
rm b.out