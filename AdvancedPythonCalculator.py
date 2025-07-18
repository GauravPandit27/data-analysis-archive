import math

class Calculator:
    def __init__(self):
        self.operations = {
            '1': ("Add", self.add),
            '2': ("Subtract", self.subtract),
            '3': ("Multiply", self.multiply),
            '4': ("Divide", self.divide),
            '5': ("Exponentiate", self.exponentiate),
            '6': ("Square Root", self.sqrt),
            '7': ("Factorial", self.factorial),
            '8': ("Logarithm", self.logarithm)
        }

    def add(self, x, y):
        return x + y

    def subtract(self, x, y):
        return x - y

    def multiply(self, x, y):
        return x * y

    def divide(self, x, y):
        if y == 0:
            raise ValueError("Cannot divide by zero.")
        return x / y

    def exponentiate(self, x, y):
        return x ** y

    def sqrt(self, x):
        if x < 0:
            raise ValueError("Cannot take the square root of a negative number.")
        return math.sqrt(x)

    def factorial(self, x):
        if x < 0:
            raise ValueError("Cannot take the factorial of a negative number.")
        return math.factorial(x)

    def logarithm(self, x, base=10):
        if x <= 0:
            raise ValueError("Logarithm is only defined for positive numbers.")
        return math.log(x, base)

    def run(self):
        while True:
            self.print_menu()
            choice = input("Enter choice: ").strip()

            if choice not in self.operations:
                print("Invalid input. Please enter a number corresponding to the operation.")
                continue

            operation_name, operation_func = self.operations[choice]

            try:
                if choice in ['1', '2', '3', '4', '5']:
                    x, y = self.get_two_numbers()
                    result = operation_func(x, y)
                elif choice == '6':
                    x = self.get_one_number()
                    result = operation_func(x)
                elif choice == '7':
                    x = self.get_integer()
                    result = operation_func(x)
                elif choice == '8':
                    x, base = self.get_number_and_base()
                    result = operation_func(x, base)

                print(f"Result of {operation_name}: {result}")
            except ValueError as e:
                print(f"Error: {e}")

            if not self.ask_continue():
                print("Thank you for using the calculator!")
                break

    def print_menu(self):
        print("\n=======================")
        print("   Python Calculator   ")
        print("=======================")
        print("Select operation:")
        for key, (name, _) in self.operations.items():
            print(f"{key}. {name}")
        print("=======================")

    def get_two_numbers(self):
        x = float(input("Enter first number: ").strip())
        y = float(input("Enter second number: ").strip())
        return x, y

    def get_one_number(self):
        x = float(input("Enter number: ").strip())
        return x

    def get_integer(self):
        x = int(input("Enter an integer: ").strip())
        return x

    def get_number_and_base(self):
        x = float(input("Enter number: ").strip())
        base_input = input("Enter logarithm base (default is 10): ").strip()
        base = float(base_input) if base_input else 10
        return x, base

    def ask_continue(self):
        return input("Do you want to perform another calculation? (yes/no): ").strip().lower() == 'yes'


if __name__ == "__main__":
    calculator = Calculator()
    calculator.run()

