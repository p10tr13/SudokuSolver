// Checks the reference CPU solver on the bundled boards and on invalid inputs.
// Usage: cpu_solver_test <boards.csv> [max_boards]
#include "../sudoku_cpu.h"

#include <cstdio>
#include <cstdlib>

static bool isValidSolution(const char* game, const char* solution)
{
	for (int i = 0; i < 81; i++)
	{
		if (solution[i] < '1' || solution[i] > '9')
			return false;
		if (game[i] != '0' && game[i] != '.' && game[i] != solution[i])
			return false; // a given digit was changed
	}
	for (int u = 0; u < 9; u++)
	{
		int row = 0, col = 0, box = 0;
		for (int k = 0; k < 9; k++)
		{
			row |= 1 << (solution[u * 9 + k] - '0');
			col |= 1 << (solution[k * 9 + u] - '0');
			box |= 1 << (solution[((u / 3) * 3 + k / 3) * 9 + (u % 3) * 3 + k % 3] - '0');
		}
		if (row != 0x3FE || col != 0x3FE || box != 0x3FE)
			return false;
	}
	return true;
}

static int expect(const char* name, int got, int want)
{
	if (got != want)
	{
		std::printf("FAIL %s: got %d, expected %d\n", name, got, want);
		return 1;
	}
	return 0;
}

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::fprintf(stderr, "usage: %s <boards.csv> [max_boards]\n", argv[0]);
		return 2;
	}
	const int maxBoards = argc > 2 ? std::atoi(argv[2]) : 10000;

	FILE* file = std::fopen(argv[1], "r");
	if (!file)
	{
		std::fprintf(stderr, "cannot open %s\n", argv[1]);
		return 2;
	}

	char line[128];
	char solution[81];
	int boards = 0, failures = 0;
	while (boards < maxBoards && std::fgets(line, sizeof line, file))
	{
		const int result = sudokuCPU(line, solution);
		if (result != 1 || !isValidSolution(line, solution))
		{
			std::printf("FAIL board %d: result %d\n", boards + 1, result);
			failures++;
		}
		boards++;
	}
	std::fclose(file);

	char empty[82], conflict[82], invalid[82];
	for (int i = 0; i < 81; i++)
		empty[i] = conflict[i] = invalid[i] = '0';
	empty[81] = conflict[81] = invalid[81] = '\0';
	conflict[0] = conflict[1] = '5';
	invalid[40] = 'x';

	failures += expect("conflicting givens", sudokuCPU(conflict, solution), -2);
	failures += expect("invalid character", sudokuCPU(invalid, solution), -1);
	failures += expect("empty board has many solutions", sudokuCPU(empty, solution), -3);

	std::printf("%d boards checked, %d failures\n", boards, failures);
	return failures == 0 ? 0 : 1;
}
