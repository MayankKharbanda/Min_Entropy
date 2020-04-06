import math

sum_digit = 0
no_of_bits = 0
with open('random_files/dev-random.bin','rb') as fr:
    byte = fr.read(1)
    while(byte):
        no_of_bits = no_of_bits + 8
        int_value = int.from_bytes(byte, 'big')
        while int_value:
            sum_digit, int_value = sum_digit + int_value % 2, int_value // 2
        byte = fr.read(1)

expectation = sum_digit / no_of_bits
print(expectation)


mean_diff_sq = (sum_digit*((1-expectation)**2)) + ((no_of_bits-sum_digit)*((0-expectation)**2))
mean_diff_cub = (sum_digit*((1-expectation)**3)) + ((no_of_bits-sum_digit)*((0-expectation)**3))
mean_diff_quad = (sum_digit*((1-expectation)**4)) + ((no_of_bits-sum_digit)*((0-expectation)**4))



standard_deviation = math.sqrt(mean_diff_sq/(no_of_bits-1))
print(standard_deviation)


skewness = (mean_diff_cub/no_of_bits)/(standard_deviation**3)
print(skewness)


kurtosis = ((mean_diff_quad/no_of_bits)/((mean_diff_sq/no_of_bits)**2))-3
print(kurtosis)