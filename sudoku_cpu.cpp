#include "sudoku_cpu.h"

#include <cstdint>
#include <cstring>

namespace
{
	constexpr int kSize = 81;
	constexpr uint16_t kAllDigits = 0x1FF; // bits 0-8 stand for digits 1-9

	struct State
	{
		uint8_t cells[kSize];   // 0 = empty, 1-9 = digit
		uint16_t rows[9];       // digits used in each row
		uint16_t cols[9];       // digits used in each column
		uint16_t boxes[9];      // digits used in each 3x3 box
	};

	int boxOf(int cell)
	{
		return (cell / 27) * 3 + (cell % 9) / 3;
	}

	void place(State& s, int cell, int digit)
	{
		const uint16_t bit = 1 << (digit - 1);
		s.cells[cell] = static_cast<uint8_t>(digit);
		s.rows[cell / 9] |= bit;
		s.cols[cell % 9] |= bit;
		s.boxes[boxOf(cell)] |= bit;
	}

	void remove(State& s, int cell, int digit)
	{
		const uint16_t bit = 1 << (digit - 1);
		s.cells[cell] = 0;
		s.rows[cell / 9] &= ~bit;
		s.cols[cell % 9] &= ~bit;
		s.boxes[boxOf(cell)] &= ~bit;
	}

	int popcount(uint16_t x)
	{
		int n = 0;
		for (; x; x &= x - 1)
			n++;
		return n;
	}

	// Backtracking that always branches on the empty cell with the fewest candidates.
	// Counts solutions up to 2 (enough to tell "unique" from "many") and keeps the first one.
	void search(State& s, int& found, uint8_t* first)
	{
		int best = -1;
		int bestCount = 10;
		uint16_t bestCandidates = 0;

		for (int cell = 0; cell < kSize; cell++)
		{
			if (s.cells[cell] != 0)
				continue;
			const uint16_t candidates = kAllDigits & ~(s.rows[cell / 9] | s.cols[cell % 9] | s.boxes[boxOf(cell)]);
			const int count = popcount(candidates);
			if (count < bestCount)
			{
				best = cell;
				bestCount = count;
				bestCandidates = candidates;
				if (count <= 1)
					break;
			}
		}

		if (best == -1) // no empty cell left: a solution
		{
			if (found == 0)
				std::memcpy(first, s.cells, kSize);
			found++;
			return;
		}

		for (int digit = 1; digit <= 9 && found < 2; digit++)
		{
			if (bestCandidates & (1 << (digit - 1)))
			{
				place(s, best, digit);
				search(s, found, first);
				remove(s, best, digit);
			}
		}
	}
}

int sudokuCPU(const char* game, char* solution)
{
	State s{};

	for (int cell = 0; cell < kSize; cell++)
	{
		const char c = game[cell];
		if (c == '0' || c == '.')
			continue;
		if (c < '1' || c > '9')
			return -1;

		const int digit = c - '0';
		const uint16_t bit = 1 << (digit - 1);
		if ((s.rows[cell / 9] | s.cols[cell % 9] | s.boxes[boxOf(cell)]) & bit)
			return -2;
		place(s, cell, digit);
	}

	int found = 0;
	uint8_t first[kSize];
	search(s, found, first);

	if (found == 0)
		return 0;

	for (int cell = 0; cell < kSize; cell++)
		solution[cell] = static_cast<char>('0' + first[cell]);

	return found == 1 ? 1 : -3;
}
