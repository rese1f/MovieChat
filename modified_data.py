with open('correctness.txt', 'r') as file:
    lines = file.readlines()

values = []
for line in lines:
    start = line.find(":")
    end = line.find("}")
    
    if start != -1 and end != -1:
        value_str = line[start + 1:end]
        
        try:
            value = float(value_str)
            values.append(value)
        except ValueError:
            print(f" {value_str}")


if values:
    average = sum(values) / len(values)
    print(f"average {average}")

