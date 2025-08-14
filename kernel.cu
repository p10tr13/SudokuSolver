#pragma comment(linker, "/STACK:16777216")

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <stdint.h>
#include <fstream>

#include <iomanip>
#include <chrono>

#include "sudokuCPU.h"

using namespace std;
using namespace std::chrono;

#define BOX_DIM 3 // Box dimension inside of sudoku

#define BOARD_DIM 9 // Board dimension

#define BOARD_SIZE 81 // Board size (9x9)

#define MAX_BOARDS 1044480 // Number of boards that can be processed by the program (for best performance would be a multiple of THREADS_IN_BLOCK * MAX_BLOCKS), if the number is bigger than number of boards in the file program will solve as many as in the file

#define MASK_ARRAY_SIZE 27 // Size of masks array (9 masks for rows, 9 masks for columns, 9 masks for boxes)

#define RECURSION_TREE_SIZE 256 // Number of boards that can be used 

#define MAX_BLOCKS 240 // Number of blocks that will be queued up for GPU

#define THREADS_IN_BLOCK 256 // Number of threads in one block

#define STACK_SIZE 100 // The size of stack for a thread, that doesn't have a free space for a new board in its GPU memory

#define INPUTFILE_PATH "sudoku.csv" // Path to the input file with sudoku boards

#define CPU_OUTPUTFILE_PATH "cpu_output.txt" // Path to the file for saving results from CPU

#define GPU_OUTPUTFILE_PATH "gpu_output.txt" // Path to the file for saving results from GPU

__host__ __device__ uint8_t cellmask_to_possibilities(uint16_t cellmask);
__host__ void print_board(char* board);
__host__ bool validateSudoku(char* board);
__host__ bool sudokuCheck(char* board);
__host__ __device__ uint8_t GetRow(uint8_t ind);
__host__ __device__ uint8_t GetCol(uint8_t ind);
__host__ __device__ uint8_t GetBox(uint8_t ind);

// Stack structure definition
typedef struct Stack
{
	uint8_t data[STACK_SIZE];
	int8_t top;
} Stack;

// Stack functions in device
__device__ void initializeStack(Stack* stack);
__device__ bool push(Stack* stack, uint8_t value);
__device__ bool pop(Stack* stack, uint8_t* value);
__device__ bool top(Stack* stack, uint8_t* value);
__device__ bool isEmptyStack(Stack* stack);
__device__ bool isFullStack(Stack* stack);

// Functions for controlArray in device
__device__ void setBit(uint8_t* controlArray, uint8_t index);
__device__ void clearBit(uint8_t* controlArray, uint8_t index);
__device__ uint8_t getBit(uint8_t* controlArray, uint8_t index);

// Sudoku solving kernel function
__global__ void solve(char* boards, uint16_t* masks, const unsigned int size);

// Function that aggregates the main CUDA calls (copying data to GPU, algorithm, reading results from GPU)
cudaError_t solvewithGPU(char* h_boards, unsigned int boards, char* d_boards, uint16_t* d_masks, long long* copy_to_d_time, long long* alg_time, long long* copy_to_h_time);

/**
 * Function that executes the sudoku solving algorithm on the GPU.
 *
 * @param boards - pointer to boards table, in which the boards are stored and there is a place for new ones.
 * @param masks - pointer to masks table, in which the masks are stored for sudoku boards and there is a place for new ones.
 * @param size - number of boards to solve in this kernel call.
 */
__global__ void solve(char* boards, uint16_t* masks, const unsigned int size)
{
	if (threadIdx.x + blockDim.x * blockIdx.x < size)
	{
		char board[BOARD_SIZE]; // Local sudoku board for the thread
		uint16_t locmasks[MASK_ARRAY_SIZE] = { 0 }; // Local mask table (masks[9 masks for rows, 9 masks for columns, 9 masks for small boxes])
		Stack numStack, indStack; // Stack for numbers and indices, where the number is or should be written (in case of backtracking when memory is full)
		uint8_t controlArray[RECURSION_TREE_SIZE / 8]; // Array for controlling the recursion tree, where each bit represents whether a board is free (0) or occupied (1) in the memory
		setBit(controlArray, 0); // Setting the first bit to 1 for the first board, which is always the initial board
		initializeStack(&numStack);
		initializeStack(&indStack);
		uint8_t i = 0; // Index of the board that we are currently solving
		uint8_t j = (i + 1) % RECURSION_TREE_SIZE; // Index of the next potential free board in the memory, that we will start looking from
		uint8_t status = 0; // Used instead of booles 1 - done, 2 - error

		// Getting the initial board
		memcpy(board, boards + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, sizeof(char) * BOARD_SIZE);

		// Mask calculation for the initial array
		for (uint8_t p = 0; p < BOARD_SIZE; p++)
		{
			uint8_t number = board[p] - '0';
			if (number != 0)
			{
				locmasks[GetRow(p)] |= 1 << (number - 1);
				locmasks[9 + GetCol(p)] |= 1 << (number - 1);
				locmasks[18 + GetBox(p)] |= 1 << (number - 1);
			}
		}

		// Saving the initial array mask
		memcpy(masks + (blockDim.x * blockIdx.x + threadIdx.x) * MASK_ARRAY_SIZE, locmasks, sizeof(uint16_t) * MASK_ARRAY_SIZE);

		while (status != 1)
		{
			// If the control number is 0 we have to find the next board to solve
			if (getBit(controlArray, i) != 1)
			{
				uint8_t old_i = i;
				i = (i + 1) % RECURSION_TREE_SIZE;
				while (getBit(controlArray, i) != 1 && old_i != i)
					i = (i + 1) % RECURSION_TREE_SIZE;
				if (old_i == i)
					break;

				memcpy(board, boards + blockDim.x * gridDim.x * BOARD_SIZE * i + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, sizeof(char) * BOARD_SIZE);
				memcpy(locmasks, masks + blockDim.x * gridDim.x * MASK_ARRAY_SIZE * i + (blockDim.x * blockIdx.x + threadIdx.x) * MASK_ARRAY_SIZE, sizeof(uint16_t) * MASK_ARRAY_SIZE);
				j = (i + 1) % RECURSION_TREE_SIZE;
			}

			uint8_t minimalpossibilities = 9;
			uint8_t minimalpossibilities_ind = 82;
			status = 1; // This status means that the board is finished. In the loop, when we encounter 0, we change this status to 0, as not finished

			// Loop over all cells in the board
			for (uint8_t k = 0; k < BOARD_SIZE; k++)
			{
				if (board[k] == '0')
				{
					status = 0; // Setting the stauts to 0, because the board is not finished yet
					uint16_t mask = (locmasks[GetRow(k)] | locmasks[GetCol(k) + 9]) | locmasks[GetBox(k) + 18]; // Sum of masks for this cell

					// If the first 9 bits are filled and there is still 0 in this cell, we have an error on the board
					if (mask == 0x01FF)
					{
						status = 2;
						break;
					}
					// If we found a cell with less possibilities than the current minimalpossibilities, we save it
					else if (minimalpossibilities > cellmask_to_possibilities(mask))
					{
						minimalpossibilities = cellmask_to_possibilities(mask);
						minimalpossibilities_ind = k;
					}
				}
			}

			// This board doesnt have a valid solution
			if (status == 2)
			{
				if (isEmptyStack(&indStack))
				{
					clearBit(controlArray, i);
				}
				else
				{
					uint8_t ind, number;
					pop(&indStack, &ind);
					pop(&numStack, &number);
					while (board[ind] != '0')
					{
						number = board[ind] - '0';
						board[ind] = '0';
						locmasks[GetRow(ind)] &= ~(1 << (number - 1));
						locmasks[GetCol(ind) + 9] &= ~(1 << (number - 1));
						locmasks[GetBox(ind) + 18] &= ~(1 << (number - 1));
						pop(&indStack, &ind);
						pop(&numStack, &number);
					}
					board[ind] = '0' + number;
					locmasks[GetRow(ind)] |= (1 << (number - 1));
					locmasks[GetCol(ind) + 9] |= (1 << (number - 1));
					locmasks[GetBox(ind) + 18] |= (1 << (number - 1));
					if (!isEmptyStack(&indStack))
					{
						push(&indStack, ind);
						push(&numStack, number);
					}
				}
			}

			// Sudoku solved
			if (status == 1)
			{
				memcpy(boards + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, board, sizeof(char) * BOARD_SIZE);
			}

			// Isn't solved yet, but we have a cell with some possibilities to fill in
			if (status == 0)
			{
				uint8_t k = 0;
				// Sum mask for the cell with minimal possibilities
				uint16_t combineMask = (locmasks[GetRow(minimalpossibilities_ind)] | locmasks[GetCol(minimalpossibilities_ind) + 9]) | locmasks[GetBox(minimalpossibilities_ind) + 18];

				// Iterating over all 9 possible digits in sudoku
				for (uint8_t l = 0; l < 9; l++)
				{
					if ((combineMask >> l) & 1) // If the digit is already used in the row, column or box, we skip it
						continue;

					if (k == minimalpossibilities - 1) // This is the last possibility for this cell 
					{
						// Store a digit in the board
						board[minimalpossibilities_ind] = l + '0' + 1;

						// Update of masks
						locmasks[GetRow(minimalpossibilities_ind)] |= (1 << l);
						locmasks[GetCol(minimalpossibilities_ind) + 9] |= (1 << l);
						locmasks[GetBox(minimalpossibilities_ind) + 18] |= (1 << l);

						if (!isEmptyStack(&numStack)) // If necessary, we push it on the stack so that we can come back later
						{
							push(&numStack, l + 1);
							push(&indStack, minimalpossibilities_ind);
						}
						break;
					}

					while (j != i && status != 5)
					{
						if (getBit(controlArray, j) == 0) // Free board found
							break;
						j = (j + 1) % RECURSION_TREE_SIZE;
					}

					if (j != i && status != 5) // Free board found
					{
						board[minimalpossibilities_ind] = l + '0' + 1;
						memcpy(boards + j * blockDim.x * gridDim.x * BOARD_SIZE + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, board, sizeof(char) * BOARD_SIZE);
						board[minimalpossibilities_ind] = '0';
						locmasks[GetRow(minimalpossibilities_ind)] |= (1 << l);
						locmasks[GetCol(minimalpossibilities_ind) + 9] |= (1 << l);
						locmasks[GetBox(minimalpossibilities_ind) + 18] |= (1 << l);
						memcpy(masks + j * blockDim.x * gridDim.x * MASK_ARRAY_SIZE + (blockDim.x * blockIdx.x + threadIdx.x) * MASK_ARRAY_SIZE, locmasks, sizeof(uint16_t) * MASK_ARRAY_SIZE);
						locmasks[GetRow(minimalpossibilities_ind)] &= ~(1 << l);
						locmasks[GetCol(minimalpossibilities_ind) + 9] &= ~(1 << l);
						locmasks[GetBox(minimalpossibilities_ind) + 18] &= ~(1 << l);
						setBit(controlArray, j);
					}

					if (j == i && status != 5) // Free board not found, so we have to push the current board to the stack
						status = 5;

					if (status == 5)
					{
						push(&numStack, l + 1);
						push(&indStack, minimalpossibilities_ind);
					}

					k++;
				}
			}
		}
	}
}

/**
 * Warming up function for GPU.
 *
 * @param i - doesn't have any meaning.
 */
__global__ void warmup(int i)
{
	int res = threadIdx.x * i;
}

int main(int argc, char** argv)
{
	FILE* inputfile;
	if (argc < 2)
		inputfile = fopen(INPUTFILE_PATH, "r");
	else
		inputfile = fopen(argv[1], "r");
	if (inputfile == NULL)
	{
		fprintf(stderr, "Cannot open file with input data.\n");
		return 1;
	}

	FILE* cpu_outputfile = fopen(CPU_OUTPUTFILE_PATH, "w");
	if (cpu_outputfile == NULL)
	{
		fprintf(stderr, "Cannot open file for saving the CPU results.\n");
		fclose(inputfile);
		return 1;
	}

	FILE* gpu_outputfile = fopen(GPU_OUTPUTFILE_PATH, "w");
	if (gpu_outputfile == NULL)
	{
		fprintf(stderr, "Cannot open file for saving the GPU results.\n");
		fclose(inputfile);
		fclose(cpu_outputfile);
		return 1;
	}

	char boards[THREADS_IN_BLOCK * MAX_BLOCKS * BOARD_SIZE]; // Table for sudoku boards read from input file of max capacity for the GPU
	char cpu_solutions[THREADS_IN_BLOCK * MAX_BLOCKS * BOARD_SIZE]; // Table for the solutions of sudoku solved by CPU
	long long cpu_time = 0, gpu_time = 0, gpu_alg_time = 0, gpu_copy_to_d_time = 0, gpu_copy_to_h_time = 0, gpu_alloc_time = 0, check_and_save_time = 0;
	int done = 0, boards_count = 0, cpu_correct_boards = 0, gpu_correct_boards = 0;
	char line[83]; // Table for the line read from input file

	// CUDA variables
	cudaError_t cudaStatus;
	char* d_boards;
	uint16_t* d_masks;

	auto gpu_alloc_ts = high_resolution_clock::now();
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	// Allocating memory for boards in GPU
	cudaStatus = cudaMalloc(&d_boards, sizeof(char) * BOARD_SIZE * MAX_BLOCKS * THREADS_IN_BLOCK * RECURSION_TREE_SIZE);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc d_boards failes!\n");
		goto Error;
	}

	// Allocating memory for masks in GPU
	cudaStatus = cudaMalloc(&d_masks, sizeof(uint16_t) * MASK_ARRAY_SIZE * MAX_BLOCKS * THREADS_IN_BLOCK * RECURSION_TREE_SIZE);
	auto gpu_alloc_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc d_masks failes!\n");
		goto Error;
	}
	gpu_alloc_time = 0.001 * duration_cast<microseconds> (gpu_alloc_te - gpu_alloc_ts).count();

	// Warmup function for GPU
	warmup <<<1, 512 >>> (2);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failes %s!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	while (fgets(line, 83, inputfile) && done + boards_count < MAX_BOARDS)
	{
		memcpy(boards + boards_count * BOARD_SIZE, line, sizeof(char) * 81);

		// Checking if the sudoku is valid and has at least 17 non-zero numbers
		if (!validateSudoku(&boards[boards_count * BOARD_SIZE]))
		{
			char zeroBoard[BOARD_SIZE] = { 0 };
			fprintf(stderr, "Invalid board\n");
			memcpy(&boards[boards_count * BOARD_SIZE], zeroBoard, sizeof(char) * BOARD_SIZE);
			continue;
		}

		boards_count++;

		if (boards_count == (THREADS_IN_BLOCK * MAX_BLOCKS)) // The batch is full
		{
			// Solving with CPU
			auto cpu_ts = high_resolution_clock::now();
			for (int i = 0; i < boards_count; i++)
			{
				int res = sudokuCPU(boards + BOARD_SIZE * i, cpu_solutions + BOARD_SIZE * i);
			}
			auto cpu_te = high_resolution_clock::now();
			cpu_time += 0.001 * duration_cast<microseconds> (cpu_te - cpu_ts).count();

			// Checking CPU solutions and saving to file
			auto cpu_check_and_save_ts_1 = high_resolution_clock::now();
			for (int i = 0; i < boards_count; i++)
			{
				if (sudokuCheck(cpu_solutions + BOARD_SIZE * i))
					cpu_correct_boards++;
				fwrite(cpu_solutions + BOARD_SIZE * i, sizeof(char), BOARD_SIZE, cpu_outputfile);
				fputc('\n', cpu_outputfile);
			}
			auto cpu_check_and_save_te_1 = high_resolution_clock::now();
			check_and_save_time += 0.001 * duration_cast<microseconds> (cpu_check_and_save_te_1 - cpu_check_and_save_ts_1).count();

			// Solving with GPU
			auto gpu_ts = high_resolution_clock::now();
			cudaStatus = solvewithGPU(boards, boards_count,d_boards, d_masks, &gpu_copy_to_d_time, &gpu_alg_time, &gpu_copy_to_h_time);
			auto gpu_te = high_resolution_clock::now();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addWithCuda failed! %s\n", cudaGetErrorString(cudaStatus));
				return 1;
			}
			gpu_time += 0.001 * duration_cast<microseconds> (gpu_te - gpu_ts).count();

			// Checking GPU solutions and saving to file
			auto gpu_check_and_save_ts_1 = high_resolution_clock::now();
			for (int i = 0; i < boards_count; i++)
			{
				if (sudokuCheck(boards + BOARD_SIZE * i))
					gpu_correct_boards++;
				fwrite(boards + BOARD_SIZE * i, sizeof(char), BOARD_SIZE, gpu_outputfile);
				fputc('\n', gpu_outputfile);
			}
			auto gpu_check_and_save_te_1 = high_resolution_clock::now();
			check_and_save_time += 0.001 * duration_cast<microseconds> (gpu_check_and_save_te_1 - gpu_check_and_save_ts_1).count();

			done += boards_count;
			printf("Solved: %d\n", done);
			boards_count = 0;
		}
	}

	// File ended, but the batch is not full, so we solve the already read and not solved boards
	if (boards_count != 0)
	{
		// Solving with CPU
		auto cpu_ts = high_resolution_clock::now();
		for (int i = 0; i < boards_count; i++)
		{
			int res = sudokuCPU(boards + BOARD_SIZE * i, cpu_solutions + BOARD_SIZE * i);
		}
		auto cpu_te = high_resolution_clock::now();
		cpu_time += 0.001 * duration_cast<microseconds> (cpu_te - cpu_ts).count();

		// Checking CPU solutions and saving to file
		auto cpu_check_and_save_ts_2 = high_resolution_clock::now();
		for (int i = 0; i < boards_count; i++)
		{
			if (sudokuCheck(cpu_solutions + BOARD_SIZE * i))
				cpu_correct_boards++;
			fwrite(cpu_solutions + BOARD_SIZE * i, sizeof(char), BOARD_SIZE, cpu_outputfile);
			fputc('\n', cpu_outputfile);
		}
		auto cpu_check_and_save_te_2 = high_resolution_clock::now();
		check_and_save_time += 0.001 * duration_cast<microseconds> (cpu_check_and_save_te_2 - cpu_check_and_save_ts_2).count();

		// Solving with GPU
		auto gpu_ts = high_resolution_clock::now();
		cudaStatus = solvewithGPU(boards, boards_count, d_boards, d_masks, &gpu_copy_to_d_time, &gpu_alg_time, &gpu_copy_to_h_time);
		auto gpu_te = high_resolution_clock::now();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "addWithCuda failed! %s\n", cudaGetErrorString(cudaStatus));
			return 1;
		}
		gpu_time += 0.001 * duration_cast<microseconds> (gpu_te - gpu_ts).count();

		// Checking GPU solutions and saving to file
		auto gpu_check_and_save_ts_2 = high_resolution_clock::now();
		for (int i = 0; i < boards_count; i++)
		{
			if (sudokuCheck(boards + BOARD_SIZE * i))
				gpu_correct_boards++;
			fwrite(boards + BOARD_SIZE * i, sizeof(char), BOARD_SIZE, gpu_outputfile);
			fputc('\n', gpu_outputfile);
		}
		auto gpu_check_and_save_te_2 = high_resolution_clock::now();
		check_and_save_time += 0.001 * duration_cast<microseconds> (gpu_check_and_save_te_2 - gpu_check_and_save_ts_2).count();

		done += boards_count;
		printf("Solved: %d\n", done);
	}

	fputc('\0', cpu_outputfile);
	fputc('\0', gpu_outputfile);

	std::cout << "CPU time:    " << setw(7) << cpu_time << " nsec" << endl;
	std::cout << "Whole GPU time:    " << setw(7) << gpu_time + gpu_alloc_time << " nsec" << endl;
	std::cout << "GPU memory alloc + SetDevice time:    " << setw(7) << gpu_alloc_time << " nsec" << endl;
	std::cout << "Copy to device GPU time:    " << setw(7) << 0.001 * gpu_copy_to_d_time << " nsec" << endl;
	std::cout << "Algorithm GPU time:    " << setw(7) << gpu_alg_time << " nsec" << endl;
	std::cout << "Copy to host GPU time:    " << setw(7) << 0.001 * gpu_copy_to_h_time << " nsec" << endl;
	std::cout << "Save to file and check time:    " << setw(7) << check_and_save_time << " nsec" << endl;
	std::cout << "CPU correct solutions: " << setw(7) << cpu_correct_boards << endl;
	std::cout << "GPU correct solutions: " << setw(7) << gpu_correct_boards << endl;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		goto Error;
	}

Error:
	fclose(inputfile);
	fclose(cpu_outputfile);
	fclose(gpu_outputfile);
	cudaFree(d_boards);
	cudaFree(d_masks);

	return 0;
}

/**
 * Helper function aggregating the main CUDA calls (copying data to GPU, algorithm, reading results from GPU).
 *
 * @param[in] h_boards - array in which are stored sudoku boards read from input file (host side).
 * @param[in] boards - number of boards to solve in this kernel call.
 * @param[in] d_boards - pointer to the array in which the boards are stored in GPU memory.
 * @param[in] d_masks - pointer to the array in which the masks are stored in GPU memory.
 * @param[out] copy_to_d_time - pointer to the variable with the time in which we copy data from host to GPU.
 * @param[out] alg_time - pointer to the variable with the time in which we run the algorithm on GPU.
 * @param[out] copy_to_h_time - pointer to the variable with the time in which we copy data from GPU to host.
 * @return error code from CUDA functions, cudaSuccess if everything went well.
 */
cudaError_t solvewithGPU(char* h_boards, unsigned int boards, char* d_boards, uint16_t* d_masks, long long* copy_to_d_time, long long* alg_time, long long* copy_to_h_time)
{
	cudaError_t cudaStatus;

	// Setting the number of blocks for the kernel function
	int blocks = 0;
	if (boards % THREADS_IN_BLOCK != 0)
		blocks = ((boards - (boards % THREADS_IN_BLOCK)) / THREADS_IN_BLOCK) + 1;
	else
		blocks = boards / THREADS_IN_BLOCK;

	// Copying boards to GPU memory
	auto gpu_memory_copy_ts = high_resolution_clock::now();
	cudaStatus = cudaMemcpy(d_boards, h_boards, sizeof(char) * BOARD_SIZE * boards, cudaMemcpyHostToDevice);
	auto gpu_memory_copy_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy d_boards failes!\n");
		goto Error;
	}
	*copy_to_d_time += duration_cast<microseconds>(gpu_memory_copy_te - gpu_memory_copy_ts).count();

	// Calling the kernel function to solve sudoku boards
	auto gpu_sol_ts = high_resolution_clock::now();
	solve <<<blocks, THREADS_IN_BLOCK>>> (d_boards, d_masks, boards);
	cudaStatus = cudaDeviceSynchronize();
	auto gpu_sol_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failes: %s!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	*alg_time += 0.001 * duration_cast<microseconds>(gpu_sol_te - gpu_sol_ts).count();

	// Copying solved boards back to host memory
	auto gpu_memory_back_ts = high_resolution_clock::now();
	cudaStatus = cudaMemcpy(h_boards, d_boards, sizeof(char) * BOARD_SIZE * boards, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failes! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	auto gpu_memory_back_te = high_resolution_clock::now();
	*copy_to_h_time += duration_cast<microseconds>(gpu_memory_back_te - gpu_memory_back_ts).count();
Error:
	return cudaStatus;
}

/**
 * Converts cellmask to the number of possibilities that can be written in the cell with this mask.
 *
 * @param cellmask - mask of the cell, where each bit represents whether a digit from 1 to 9 can be written in the cell (0 - can be written, 1 - cannot be written).
 * @return Possibilities that can be written in the cell with this mask (0-9).
 */
__host__ __device__ uint8_t cellmask_to_possibilities(uint16_t cellmask)
{
	uint8_t result = 0;

	for (uint8_t i = 0; i < 9; i++)
	{
		result += ~(cellmask) & 1;
		cellmask >>= 1;
	}

	return result;
}

/**
 * Displays the sudoku board in a readable format.
 *
 * @param board - pointer to the board to be displayed.
 */
__host__ void print_board(char* board)
{
	for (int i = 0; i < 9; ++i)
	{
		if (i % 3 == 0 && i != 0)
		{
			printf("----------------------\n");
		}

		for (int j = 0; j < 9; ++j)
		{
			if (j % 3 == 0 && j != 0)
			{
				printf("| ");
			}
			printf("%c ", board[i * 9 + j]);
		}
		printf("\n");
	}
}

/**
 * Checks if the sudoku board is valid (1 solution and no contradictions).
 *
 * @param board - pointer to the board to be checked.
 * 
 * @returns Validation result: true if the board is valid and has at least 17 non-zero numbers, false otherwise.
 */
__host__ bool validateSudoku(char* board)
{
	int not_zeroes_count = 0;
	uint16_t masks[MASK_ARRAY_SIZE] = { 0 };

	for (int i = 0; i < BOARD_SIZE; i++)
	{
		int num = board[i] - '0';

		if ((masks[GetRow(i)] >> (num - 1)) & 1 || (masks[9 + GetCol(i)] >> (num - 1)) & 1 || (masks[18 + GetBox(i)] >> (num - 1)) & 1)
			return false;

		masks[GetRow(i)] |= 1 << (num - 1);
		masks[9 + GetCol(i)] |= 1 << (num - 1);
		masks[18 + GetBox(i)] |= 1 << (num - 1);
		not_zeroes_count++;
	}

	if (not_zeroes_count < 17)
		return false;
	return true;
}

/**
 * Checks solved sudoku board for correctness.
 *
 * @param board - pointer to the board to be checked.
 *
 * @returns correctness of the board: true if the board is correct, false otherwise.
 */
__host__ bool sudokuCheck(char* board)
{
	uint16_t masks[MASK_ARRAY_SIZE] = { 0 };
	for (int i = 0; i < BOARD_SIZE; i++)
	{
		int num = board[i] - '0';
		if (num < 1 || num > 9)
			return false;
		else
		{
			if ((masks[GetRow(i)] >> (num - 1)) & 1 || (masks[9 + GetCol(i)] >> (num - 1)) & 1 || (masks[18 + GetBox(i)] >> (num - 1)) & 1)
				return false;
			masks[GetRow(i)] |= 1 << (num - 1);
			masks[9 + GetCol(i)] |= 1 << (num - 1);
			masks[18 + GetBox(i)] |= 1 << (num - 1);
		}
	}
	return true;
}

/**
 * Converts the index of a cell in the sudoku board to the row number.
 *
 * @param ind - index of the cell.
 * @return row number where the cell is located (counted from 0).
 */
__host__ __device__ uint8_t GetRow(uint8_t ind)
{
	return (ind - GetCol(ind)) / BOARD_DIM;
}

/**
 * Converts the index of a cell in the sudoku board to the column number.
 *
 * @param ind - index of the cell.
 * @return column number where the cell is located (counted from 0).
 */
__host__ __device__ uint8_t GetCol(uint8_t ind)
{
	return ind % BOARD_DIM;
}

/**
 * Converts the index of a cell in the sudoku board to the box number.
 *
 * @param ind - index of the cell.
 * @return number of the box where the cell is located (counted from 0, left upper corner).
 */
__host__ __device__ uint8_t GetBox(uint8_t ind)
{
	uint8_t row = GetRow(ind);
	uint8_t col = GetCol(ind);
	uint8_t boxrow = row - (row % BOX_DIM);
	uint8_t boxcol = (col - (col % BOX_DIM)) / BOX_DIM;
	return boxrow + boxcol;
}

/**
 * Initializes the stack by setting the top index to -1, indicating that the stack is empty.
 *
 * @param stack - pointer to the stack to be initialized.
 */
__device__ void initializeStack(Stack* stack)
{
	stack->top = -1;
}

/**
 * Pushes a value onto the top of the stack.
 *
 * @param stack - pointer to the stack.
 * @param value - value to be pushed onto the stack (0-255).
 * @return information about the success of the action, returns false if the stack is full.
 */
__device__ bool push(Stack* stack, uint8_t value)
{
	if (isFullStack(stack))
	{
		printf("Error: Stack if full!\n");
		return false;
	}
	stack->data[++stack->top] = value;
	return true;
}

/**
 * Pops a value from the top of the stack.
 *
 * @param[in] stack - pointer to the stack.
 * @param[out] value -  pointer to the value that is popped from the stack.
 * @return information about the success of the action, returns false if the stack is empty.
 */
__device__ bool pop(Stack* stack, uint8_t* value)
{
	if (isEmptyStack(stack))
	{
		printf("Error: Stack is empty!\n");
		return false;
	}
	*value = stack->data[stack->top--];
	return true;
}

/**
 * Peeks at the value on the top of the stack without removing it.
 *
 * @param[in] stack - pointer to the stack.
 * @param[out] value -  pointer to the value that is on the top of the stack.
 * @return information about the success of the action, returns false if the stack is empty.
 */
__device__ bool top(Stack* stack, uint8_t* value)
{
	if (isEmptyStack(stack))
	{
		printf("Error: Stack is empty!\n");
		return false;
	}
	*value = stack->data[stack->top];
	return true;
}

/**
 * Checks if the stack is empty.
 *
 * @param stack - poiter to the stack.
 * @return information about the emptiness of the stack, returns true if the stack is empty, false otherwise.
 */
__device__ bool isEmptyStack(Stack* stack)
{
	return stack->top == -1;
}

/**
 * Checks if the stack is full.
 *
 * @param stack - pointer to the stack.
 * @return information about the fullness of the stack, returns true if the stack is full, false otherwise.
 */
__device__ bool isFullStack(Stack* stack)
{
	return stack->top == STACK_SIZE - 1;
}

/**
 * Sets a bit at the index specified to 1.
 *
 * @param controlArray - pointer to the array in which we change the value of bit.
 * @param index - index of the bit in the array to be set to 1.
 */
__device__ void setBit(uint8_t* controlArray, uint8_t index)
{
	if (index < 0 || index >= RECURSION_TREE_SIZE)
		printf("Wrong index!\n");
	else
		controlArray[index / 8] |= (1 << (index % 8));
}

/**
 * Clears a bit at the index specified, setting it to 0.
 *
 * @param controlArray - pointer to the array in which we change the value of bit.
 * @param index - index of the bit in the array to be cleared (set to 0).
 */
__device__ void clearBit(uint8_t* controlArray, uint8_t index)
{
	if (index < 0 || index >= RECURSION_TREE_SIZE)
		printf("Wrong index!\n");
	else
		controlArray[index / 8] &= ~(1 << (index % 8));
}

/**
 * Gets a bit value at the specified index in the array.
 *
 * @param controlArray - pointer to the array from which we read the bit value.
 * @param index - index of the bit in the array to be read (0-RECURSION_TREE_SIZE-1).
 *
 * @returns Bit value at the specified index: 0 if the bit is not set, 1 if the bit is set, or 3 if the index is out of bounds.
 */
__device__ uint8_t getBit(uint8_t* controlArray, uint8_t index) {
	if (index < 0 || index >= RECURSION_TREE_SIZE)
	{
		printf("Wrong index!\n");
		return 3;
	}
	return (controlArray[index / 8] & (1 << (index % 8))) != 0;
}
