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





lag_byte = 2

with open('random_files/dev-random.bin','rb') as fr:
    
    byte_start = fr.read(1)
    
    fr.seek(lag_byte, 1)
    byte_lag = fr.read(1)
    forward = 1
    
    partial_sum = 0

    
    while(byte_start and byte_lag):
        
        
        byte_start_string = format(int.from_bytes(byte_start, 'big'),'08b')
        byte_lag_string = format(int.from_bytes(byte_lag, 'big'),'08b')
        
        for i in range(8):
            bit_1, bit_2 = int(byte_start_string[i]), int(byte_lag_string[i])
            partial_sum = partial_sum + ((bit_1-expectation)*(bit_2-expectation))
        
        
        if(forward == 0):
            byte_start = fr.read(1)
    
            fr.seek(lag_byte, 1)
            byte_lag = fr.read(1)
            forward = 1
    
        else:
            byte_lag = fr.read(1)
            
            fr.seek(-lag_byte, 1)
            byte_start = fr.read(1)
            forward = 0
            


auto_correlation = partial_sum/mean_diff_sq
print(auto_correlation)

norm_auto_correlation = (auto_correlation-expectation)/standard_deviation
print(norm_auto_correlation)