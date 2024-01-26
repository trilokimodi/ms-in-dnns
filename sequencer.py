import argparse
import sys


def parse_args_ass1(args):
    if args.sequence == "fibonacci":
        fibonacciList = [0, 1]
        if args.length > 2:
            for i in range(2, args.length):
                fibonacciList.append(fibonacciList[i - 2] + fibonacciList[i - 1])
        return fibonacciList

    elif args.sequence == "prime":
        primeList = [2, 3]
        isPrime = 4
        if args.length > 2:
            while len(primeList) < args.length:
                divisible = 2
                divisibleFlag = False
                for i in range(divisible, (int)((isPrime / 2) + 1)):
                    if isPrime % divisible == 0:
                        isPrime += 1
                        divisibleFlag = True
                        break
                    else:
                        divisible += 1
                if divisibleFlag is False:
                    primeList.append(isPrime)
                    isPrime += 1
        return primeList

    elif args.sequence == "square":
        squaresList = []
        if args.length > 0:
            for i in range(1, args.length + 1):
                squaresList.append(i * i)
        return squaresList

    elif args.sequence == "triangular":
        traingularList = []
        for i in range(args.length):
            traingularList.append(((i + 1) * (i + 2)) / 2)
        return traingularList

    elif args.sequence == "factorial":
        factorialList = [1, 2]
        if args.length > 2:
            for i in range(3, args.length + 1):
                j = 2
                fac = 1
                while j <= i:
                    fac *= j
                    j += 1
                factorialList.append(fac)
        return factorialList

    else:
        print("invalid choice", file=sys.stderr)


# Output:
# This is an error message


def main(args):
    return parse_args_ass1(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--sequence")
    args = parser.parse_args()

    main(args)
