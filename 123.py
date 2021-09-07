
# Python program explaining
# argpartition() function
   
import numpy as geek
  
# input array
in_arr = geek.array([ 2, 0,  1, 5, 4, 1, 9])
print ("Input unsorted array : ", in_arr) 
  
out_arr = geek.argsort(in_arr)
print ("Output sorted array indices : ", out_arr)
print("Output sorted array : ", in_arr[out_arr])