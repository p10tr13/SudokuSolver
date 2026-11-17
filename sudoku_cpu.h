#pragma once

// Reference sequential Sudoku solver used to check and time the GPU solver.
//
// game:     81 characters, '1'-'9' for given digits, '0' or '.' for empty cells.
// solution: buffer for 81 characters; receives the solved board ('1'-'9').
//
// Returns:
//    1 : exactly one solution found (written to `solution`)
//    0 : no solution
//   -1 : invalid character in the input
//   -2 : conflict between the given digits
//   -3 : more than one solution (the first one found is written to `solution`)
int sudokuCPU(const char* game, char* solution);
