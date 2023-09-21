import time


def main():
	pass


if __name__ == "__main__":
	t_start = time.time()
	main()
	exe_time = time.time() - t_start
	print("Execution time: {} [s]".format(round(exe_time,5)))

