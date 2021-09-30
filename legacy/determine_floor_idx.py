"""Question:
		Given K balls, N floors of a building, identify at which floor (i.e., the floor index) a ball breaks.

	Solution:
		Dynamic programming (see determine_floor_idx())

	Instruction:
		python3 determine_floor_idx.py
"""

# Email: kun.bj@outlook.com
# Author: kun
# Date: 2021-08-27

MAX_INT = 10 * 100  # The maximum value for K and N. If you want, you can change it.


def determine_floor_idx(K, N):
	""" Solution: Dynamic programming
		1) Create a 2 dimensional array dp[][],
			where dp[i][j] means that the maximum number of floor that I can check when I have j balls and i trials.
		2) The dp equation is: dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j] + 1,
			which means I can take 1 move to a floor (i.e., 1 trial),
			if the ball breaks, then I can check dp[i - 1][j - 1] floors.
			if the ball doesn't breaks, then I can check dp[i - 1][j] floors.

		Time complexity: O(K*N)
		Space complexity: O(K*N)

	input:
		K: int
			The number of balls.
		N: int
			The number of floors of a building.

	Returns
	-------
		i: int
			The minimum number of trials to identify the floor index
	"""
	# Edge cases
	if not K or not N:
		return 0
	if type(K) != int or type(N) != int:
		return 0
	if K <= 0 or N <= 0:
		return 0
	if K > MAX_INT or N > MAX_INT:
		return 0

	dp = [[0] * (K + 1) for _ in range(N + 1)]
	for i in range(1, N + 1):
		for j in range(1, K + 1):
			dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j] + 1
		if dp[i][K] >= N:
			return i
	return 0


def main():
	""" Test the function on different cases

	Returns
	-------

	"""
	testing_cases = [
		# edge cases
		[0, 100],  # K, N
		[1, 100],
		[100, 0],
		[100, 1],
		[0, 0],
		[-1, ],
		[],
		[-1, 10 * 100 + 1],
		[1, 2],
		[10 * 100, 100],
		[10 * 100, 100, 10],
		['', ''],
		['a', 1],
		[1.5, 2.0],

		# useful cases
		[2, 100],
	]
	tot = len(testing_cases)
	for i, vs in enumerate(testing_cases):
		if len(vs) != 2:
			print(f'({i + 1}/{tot}), Test_{i + 1}: {vs} is invalid, skip it!')
			continue
		K, N = vs[0], vs[1]
		res = determine_floor_idx(K, N)
		print(f'({i + 1}/{tot}), Test_{i + 1}: Given K ({K}) and N ({N}), the floor index is {res}.')


if __name__ == '__main__':
	main()
