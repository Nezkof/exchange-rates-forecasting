import math

def __calculate_by_table(x):
   if x == 0:
      return 1
   elif x in {1, 2, 3, 4, 6, 7, 8, 9}:
      return 0.46 * x
   elif x in {10, 20, 30, 40, 60, 80, 90}:
      return (0.357 - 0.00163 * x) * x
   elif x in {35, 45, 55, 65, 75, 85, 95}:
      return (0.213 - 0.00067 * x) * x
   elif x == 5:
      return 2.8
   elif x == 15:
      return 6.48
   elif x == 25:
      return 6.75
   elif x == 50:
      return 24
   elif 10 <= x <= 99:
      left = math.floor(x / 10) * 10 + 5
      right = x - math.floor(x / 10) * 10
      return 0.5 * (get_transition_points_distance(left) + get_transition_points_distance(right))
   else:
      raise ValueError(f"Немає правила для x = {x}")

def get_transition_points_distance(T):
   T = abs(round(T))
   
   if T <= 99:
        return __calculate_by_table(T)

   digits = [int(d) for d in str(T)][::-1]  
   q = 0
   while q < len(digits) and digits[q] == 0:
      q += 1

   if q >= len(digits):
      raise ValueError("Некоректне число T")
      
   q_temp = q + 1

   rq = digits[q]
   r = q_temp % 3

   if r == 0:  # Клас M0
        x = rq * 10
        return __calculate_by_table(x) * 10 ** (q_temp - 2)

   elif r == 1:  # Клас M1
        if q + 1 >= len(digits) or digits[q + 1] == 0:
            x = rq
        else:
            rq1 = digits[q + 1]
            x = rq1 * 10 + rq
        return __calculate_by_table(x) * 10 ** (q_temp - 1)

   elif r == 2:  # Клас M2
        if q + 1 >= len(digits) or digits[q + 1] == 0:
            x = rq * 10
        else:
            rq1 = digits[q + 1]
            x = rq1 * 10 + rq
        return __calculate_by_table(x) * 10 ** (q_temp - 1)

   else:
        raise ValueError(f"Невідомий залишок r = {r}")

