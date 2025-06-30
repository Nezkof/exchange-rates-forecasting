def use_vector_sum(arr1, arr2):
   sums = []

   for i in range(len(arr1)):
      sum = arr1[i] + arr2[i]
      sums.append(sum)

   return sums 

def use_vector_multiplication(arr1, arr2):
   result = []   
   
   for i in range(len(arr1)):
      result.append(arr1[i] * arr2[i])
   
   return result