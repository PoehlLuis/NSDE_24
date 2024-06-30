#%%
# First we let the user "input" be done throught putting numbers in the variables
# Later on we might do it throught the cmd or a pop-up-window
import numpy as np
import boundary_condition as bc

# used for testing the program, just wirte the values in here
def getGeometryInputs_hard_coded():
    width, height, order_num_int, amount_of_nodes_per_axis = 90, 180, 4, 10

    return width, height, order_num_int, amount_of_nodes_per_axis

# for finished version, lets user input values and error-checks them
def getGeometryInputs():
    width, height, order_num_int, amount_of_nodes_per_axis = 0, 0, 0, 0

    print("Input the Width: ")
    width = getNumberAboveZero()
    print("Input the Height: ")
    height = getNumberAboveZero()
    print("Input the Order of the Numerical Integration: ")
    order_num_int = getNumberFromUserInRange(4)
    print("Input the Amount of Nodes per Axis: ")
    amount_of_nodes_per_axis = getIntegerNumberAboveZero()

    return width, height, order_num_int, amount_of_nodes_per_axis

# get a non-integer number above zero from the user throught the console and makes error checks
def getNumberAboveZero():
    while True:
        number = input()
        try:
            number = float(number)
            if number > 0:
                return number
            else:
                print("Please enter a number greater than 0.\n")
        except ValueError:
            print("Please enter a number greater than 0.\n")

# get a non-integer number (can also be zero) from the user throught the console and makes error checks
def getNumberWithZero():
    while True:
        number = input()
        try:
            number = float(number)
            return number
        except ValueError:
            print("Please enter a number greater than 0.\n")


# get a integer number above zero from the user throught the console and makes error checks
def getIntegerNumberAboveZero():
    while(1):
        number = input()
        if(number.isnumeric()):  # checks if value is numeric
            if(int(number) > 0):  # checks if numeric value is greater than 0
                return int(number)
        print("Please enter a Number greater than 0.\n")

# get a number in a specific range from the user throught the console and makes error checks
def getNumberFromUserInRange(range):
    while(1):
        number = input()
        if(number.isnumeric()):  # checks if value is numeric
            if(int(number) > 0):  # checks if numeric value is greater than 0 and less than thr given range
                if(int(number) <= range):
                    return int(number)
        print("Please enter a Number greater than 0 and less (or equal) than " + str(range) + ".\n")

def getNumberFromUserInRangeWithZero(range):
    while(1):
        number = input()
        if(number.isnumeric()):  # checks if value is numeric
            if(int(number) >= 0):  # checks if numeric value is greater|equal  0 and less than thr given range
                if(int(number) <= range):
                    return int(number)
        print("Please enter a Number greater than 0 and less than " + str(range) + ".\n")

def getNumberFromUserWithAll():
    while(1):
        number = input()
        if(number.isnumeric()):  # checks if value is numeric
            return int(number)
        print("Please enter a Number.\n")

# user input for line with given value
def getLineInputs(width, height):
    start, end, value_function, amount_of_line_points = 0, 0, "", 0

    print("X-Coordinate of the Start-Point of the Line: ")
    x_start = getNumberFromUserInRangeWithZero(width)
    print("Y-Coordinate of the Start-Point of the Line: ")
    y_start = getNumberFromUserInRangeWithZero(height)
    print("X-Coordinate of the End-Point of the Line: ")
    x_end = getNumberFromUserInRangeWithZero(width)
    print("Y-Coordinate of the End-Point of the Line: ")
    y_end = getNumberFromUserInRangeWithZero(height)
    print("Input the value of the Line: ")
    value_function = input()
    print("Input the amount of Points the line should be interpolated by: ")
    amount_of_line_points = getIntegerNumberAboveZero()

    start = (x_start, x_end)
    end = (y_start, y_end)

    return start, end, value_function, amount_of_line_points

def getLineInputs_hard_coded(height, width):
    start = (50, 180)
    end = (50, 0)
    value_function = "0"
    amount_of_line_points = 10

    #start = (10, 160)
    #end = (57, 100)
    #value_function = "{x}+{y}"

    return start, end, value_function, amount_of_line_points


def getMaterialTensor():
    print("First value of the material Tensor: ")
    value1 = getNumberWithZero()
    print("Second value of the material Tensor: ")
    value2 = getNumberWithZero()
    print("Third value of the material Tensor: ")
    value3 = getNumberWithZero()
    print("Fourth value of the material Tensor: ")
    value4 = getNumberWithZero()
    return np.array([[value1, value2],[value3, value4]])

def getMaterialTensor_hard_coded():
    return np.array([[1, 1], [1, 1]])

def getBCInputs():
    left_side = getBCInput("left")
    top_side = getBCInput("top")
    right_side = getBCInput("right")
    bottom_side = getBCInput("bottom")
    return left_side, top_side, right_side, bottom_side

def getBCInput(side):
    bc = [0,0]
    print(f"Input the {side} boundary type  0=Dirichlet 1=Neumann: ")
    bc[0] = getNumberFromUserInRangeWithZero(1)
    print(f"Input the {side} boundary value: ")
    bc[1] = getNumberFromUserWithAll()
    return bc

# value 10 on the left, value 0 on the right, neumann 0 on top an bottom
def getBCInputs_hard_coded():
    return [bc.BoundaryCondition(10,"Dirichlet"),bc.BoundaryCondition(0,"Dirichlet"),bc.BoundaryCondition(0,"Dirichlet"),bc.BoundaryCondition(0,"Dirichlet")]