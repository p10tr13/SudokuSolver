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

#define BOX_DIM 3

#define BOARD_DIM 9 // BOX_DIM * BOX_DIM

#define BOARD_SIZE 81 // BOARD_DIM * BOARD_DIM

#define MAX_BOARDS 614400

#define MASK_ARRAY_SIZE 27

#define RECURSION_TREE_SIZE 256

#define MAX_BLOCKS 60

#define THREADS_IN_BLOCK 256

#define STACK_SIZE 40

__host__ __device__ uint8_t cellmask_to_possibilities(uint16_t cellmask);
__host__ void print_board(char* board);
__host__ bool setMasks(uint16_t* masks, char* board, int i);
__host__ __device__ uint8_t GetRow(uint8_t ind);
__host__ __device__ uint8_t GetCol(uint8_t ind);
__host__ __device__ uint8_t GetBox(uint8_t ind);

// Stack functions and structure
typedef struct Stack
{
	uint8_t data[STACK_SIZE];
	int8_t top;
} Stack;

__device__ void initializeStack(Stack* stack);
__device__ bool push(Stack* stack, uint8_t value);
__device__ bool pop(Stack* stack, uint8_t* value);
__device__ bool top(Stack* stack, uint8_t* value);
__device__ bool isEmptyStack(Stack* stack);
__device__ bool isFullStack(Stack* stack);

__global__ void solve(char* boards, uint16_t* masks, const unsigned int size, const unsigned int max_rec_size);
__global__ void solveglob(char* boards, uint16_t* masks, char* controlArray, unsigned int size, unsigned int max_rec_size);

cudaError_t solvewithGPU(char* h_boards, uint16_t* h_masks, unsigned int boards, long long* copy_to_d_time, long long* alg_time, long long* copy_to_h_time);

__global__ void solve(char* boards, uint16_t* masks, const unsigned int size, const unsigned int max_rec_size)
{
	char board[BOARD_SIZE];
	uint16_t locmasks[MASK_ARRAY_SIZE];
	Stack numStack, indStack;
	char controlArray[RECURSION_TREE_SIZE];
	controlArray[0] = 1;
	initializeStack(&numStack);
	initializeStack(&indStack);
	uint8_t i = 0;
	uint8_t j = (i + 1) % max_rec_size;
	uint8_t status = 0; // To jest zamiast booli 1 - done rozpatrywanej tablicy, 2 - error rozpatrywanej tablicy

	memcpy(board, boards + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, sizeof(char) * BOARD_SIZE);
	memcpy(locmasks, masks + (blockDim.x * blockIdx.x + threadIdx.x) * MASK_ARRAY_SIZE, sizeof(uint16_t) * MASK_ARRAY_SIZE);

	while (status != 1)
	{
		if (controlArray[i] != 1)
		{
			while (controlArray[i] != 1)
				i = (i + 1) % max_rec_size;
			memcpy(board, boards + blockDim.x * gridDim.x * BOARD_SIZE * i + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, sizeof(char) * BOARD_SIZE);
			memcpy(locmasks, masks + blockDim.x * gridDim.x * MASK_ARRAY_SIZE * i + (blockDim.x * blockIdx.x + threadIdx.x) * MASK_ARRAY_SIZE, sizeof(uint16_t) * MASK_ARRAY_SIZE);
			j = (i + 1) % max_rec_size;
		}

		uint8_t minimalpossibilities = 9;
		uint8_t minimalpossibilities_ind = 82;
		status = 1;

		for (uint8_t k = 0; k < BOARD_SIZE; k++)
		{
			if (board[k] == '0')
			{
				status = 0;
				uint16_t mask = (locmasks[GetRow(k)] | locmasks[GetCol(k) + 9]) | locmasks[GetBox(k) + 18];
				if (mask == 0x01FF)
				{
					status = 2;
					break;
				}
				else if (minimalpossibilities > cellmask_to_possibilities(mask))
				{
					minimalpossibilities = cellmask_to_possibilities(mask);
					minimalpossibilities_ind = k;
				}
			}
		}

		// Ta plansza nie ma rozwiązania
		if (status == 2)
		{
			if (isEmptyStack(&numStack))
			{
				controlArray[i] = 0;
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
				if (!isEmptyStack(&numStack))
				{
					push(&indStack, ind);
					push(&numStack, number);
				}
			}
		}

		// Udało się rozwiązać sudoku
		if (status == 1)
		{
			memcpy(boards + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, board, sizeof(char) * BOARD_SIZE);
		}

		if (status == 0)
		{
			uint8_t k = 0;
			uint16_t combineMask = (locmasks[GetRow(minimalpossibilities_ind)] | locmasks[GetCol(minimalpossibilities_ind) + 9]) | locmasks[GetBox(minimalpossibilities_ind) + 18];

			for (uint8_t l = 0; l < 9; l++)
			{
				if ((combineMask >> l) & 1)
					continue;

				if (k == minimalpossibilities - 1)
				{
					board[minimalpossibilities_ind] = l + '0' + 1;
					locmasks[GetRow(minimalpossibilities_ind)] |= (1 << l);
					locmasks[GetCol(minimalpossibilities_ind) + 9] |= (1 << l);
					locmasks[GetBox(minimalpossibilities_ind) + 18] |= (1 << l);
					if (!isEmptyStack(&numStack))
					{
						push(&numStack, l + 1);
						push(&indStack, minimalpossibilities_ind);
					}
					break;
				}

				while (j != i && status != 5)
				{
					if (controlArray[j] == 0)
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
						controlArray[j] = 1;
						break;
					}
					j = (j + 1) % max_rec_size;
				}

				if (j == i && status != 5)
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

__global__ void solveglob(char* boards, uint16_t* masks, char* controlArray, unsigned int size, unsigned int max_rec_size)
{
	int globind = blockDim.x * blockIdx.x + threadIdx.x;
	int bind = globind * BOARD_SIZE;
	int mind = globind * MASK_ARRAY_SIZE;
	int boards_shift = blockDim.x * gridDim.x * BOARD_SIZE;
	int masks_shift = blockDim.x * gridDim.x * MASK_ARRAY_SIZE;
	bool workFlag = true;
	Stack numStack, indStack;
	initializeStack(&numStack);
	initializeStack(&indStack);

	while (workFlag)
	{
		for (int i = 0; i < max_rec_size && workFlag; i++)
		{
			int j = (i + 1) % max_rec_size;

			while (controlArray[globind + blockDim.x * gridDim.x * i] == 1 && workFlag)
			{
				int minimalpossibilities = 9;
				int minimalpossibilities_ind = 82;
				bool error = false;
				bool doneFlag = true;

				for (int k = 0; k < BOARD_SIZE; k++)
				{
					if (boards[bind + boards_shift * i + k] == '0')
					{
						doneFlag = false;
						uint16_t mask = (masks[masks_shift * i + mind + GetRow(k)] | masks[masks_shift * i + mind + GetCol(k) + 9]) | masks[masks_shift * i + mind + GetBox(k) + 18];
						if (mask == 0x01FF)
						{
							error = true;
							break;
						}
						else if (minimalpossibilities > cellmask_to_possibilities(mask))
						{
							minimalpossibilities = cellmask_to_possibilities(mask);
							minimalpossibilities_ind = k;
						}
					}
				}

				if (error)
				{
					if (isEmptyStack(&numStack))
					{
						controlArray[globind + blockDim.x * gridDim.x * i] = 0;
					}
					else
					{
						uint8_t ind, number;
						pop(&indStack, &ind);
						pop(&numStack, &number);
						while (boards[bind + boards_shift * i + ind] != '0')
						{
							number = boards[bind + boards_shift * i + ind] - '0';
							boards[bind + boards_shift * i + ind] = '0';
							masks[mind + masks_shift * i + GetRow(ind)] &= ~(1 << (number - 1));
							masks[mind + masks_shift * i + GetCol(ind) + 9] &= ~(1 << (number - 1));
							masks[mind + masks_shift * i + GetBox(ind) + 18] &= ~(1 << (number - 1));
							pop(&indStack, &ind);
							pop(&numStack, &number);
						}
						boards[bind + boards_shift * i + ind] = '0' + number;
						masks[mind + masks_shift * i + GetRow(ind)] |= (1 << (number - 1));
						masks[mind + masks_shift * i + GetCol(ind) + 9] |= (1 << (number - 1));
						masks[mind + masks_shift * i + GetBox(ind) + 18] |= (1 << (number - 1));
						if (!isEmptyStack(&numStack))
						{
							push(&indStack, ind);
							push(&numStack, number);
						}
					}
				}

				if (doneFlag && !error)
				{
					workFlag = false;
					memcpy(boards + bind, boards + bind + boards_shift * i, sizeof(char) * BOARD_SIZE);
				}

				if (!doneFlag && !error && workFlag)
				{
					int k = 0;
					uint16_t oldRowMask = masks[mind + masks_shift * i + GetRow(minimalpossibilities_ind)];
					uint16_t oldColMask = masks[mind + masks_shift * i + GetCol(minimalpossibilities_ind) + 9];
					uint16_t oldBoxMask = masks[mind + masks_shift * i + GetBox(minimalpossibilities_ind) + 18];
					uint16_t combineMask = (oldRowMask | oldColMask) | oldBoxMask;

					bool full = false;

					for (int l = 0; l < 9; l++)
					{
						if ((combineMask >> l) & 1)
							continue;

						if (k == minimalpossibilities - 1)
						{
							boards[bind + boards_shift * i + minimalpossibilities_ind] = l + '0' + 1;
							masks[mind + masks_shift * i + GetRow(minimalpossibilities_ind)] = oldRowMask | (1 << l);
							masks[mind + masks_shift * i + GetCol(minimalpossibilities_ind) + 9] = oldColMask | (1 << l);
							masks[mind + masks_shift * i + GetBox(minimalpossibilities_ind) + 18] = oldBoxMask | (1 << l);
							if (!isEmptyStack(&numStack))
							{
								push(&numStack, l + 1);
								push(&indStack, minimalpossibilities_ind);
							}
							break;
						}

						while (j != i && !full)
						{
							if (controlArray[globind + j * blockDim.x * gridDim.x] == 0)
							{
								memcpy(boards + j * boards_shift + bind, boards + bind + boards_shift * i, sizeof(char) * BOARD_SIZE);
								memcpy(masks + j * masks_shift + mind, masks + mind + masks_shift * i, sizeof(uint16_t) * MASK_ARRAY_SIZE);
								boards[j * boards_shift + bind + minimalpossibilities_ind] = l + '0' + 1;
								masks[j * masks_shift + mind + GetRow(minimalpossibilities_ind)] = oldRowMask | (1 << l);
								masks[j * masks_shift + mind + GetCol(minimalpossibilities_ind) + 9] = oldColMask | (1 << l);
								masks[j * masks_shift + mind + GetBox(minimalpossibilities_ind) + 18] = oldBoxMask | (1 << l);
								controlArray[globind + j * blockDim.x * gridDim.x] = 1;
								break;
							}
							j = (j + 1) % max_rec_size;
						}

						if (j == i && !full)
							full = true;

						if (full)
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
}

__global__ void rozgrzewka(int i)
{
	int res = threadIdx.x * i;
}

int main()
{
	FILE* inputfile = fopen("C:/Users/Piotr/Desktop/Sudoku/sudoku.csv", "r");
	if (inputfile == NULL)
	{
		fprintf(stderr, "Nie można otworzyć pliku z danymi\n");
		return 1;
	}

	FILE* cpu_outputfile = fopen("C:/Users/Piotr/Desktop/Sudoku/cpu_output.txt", "w");
	if (cpu_outputfile == NULL)
	{
		fprintf(stderr, "Nie można otworzyć pliku do zapisu wyników z cpu\n");
		return 1;
	}

	FILE* gpu_outputfile = fopen("C:/Users/Piotr/Desktop/Sudoku/gpu_output.txt", "w");
	if (gpu_outputfile == NULL)
	{
		fprintf(stderr, "Nie można otworzyć pliku do zapisu wyników z gpu\n");
		return 1;
	}

	char boards[THREADS_IN_BLOCK * MAX_BLOCKS * BOARD_SIZE];
	uint16_t masks[MASK_ARRAY_SIZE * THREADS_IN_BLOCK * MAX_BLOCKS];
	int boards_count = 0;
	char cpu_solutions[THREADS_IN_BLOCK * MAX_BLOCKS * BOARD_SIZE];
	long long cpu_time = 0, gpu_time = 0, gpu_alg_time = 0, gpu_copy_to_d_time = 0, gpu_copy_to_h_time = 0;
	int done = 0;
	char line[83];
	cudaError_t cudaStatus;

	// Część rozgrzewająca GPU (ta funkcja nic nie robi)
	rozgrzewka <<<1, 512 >>> (2);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failes %s!\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	while (fgets(line, 83, inputfile) && done + boards_count < MAX_BOARDS)
	{
		memcpy(boards + boards_count * BOARD_SIZE, line, sizeof(char) * 81);
	
		if (!setMasks(masks, &boards[boards_count * BOARD_SIZE], boards_count))
		{
			char zeroBoard[BOARD_SIZE] = { 0 };
			fprintf(stderr, "Invalid board\n");
			memcpy(&boards[boards_count * BOARD_SIZE], zeroBoard, sizeof(char) * BOARD_SIZE);
			continue;
		}
	
		boards_count++;
	
		if (boards_count == (THREADS_IN_BLOCK * MAX_BLOCKS))
		{
			auto cpu_ts = high_resolution_clock::now();
			for (int i = 0; i < boards_count; i++)
			{
				int res = sudokuCPU(boards + BOARD_SIZE * i, cpu_solutions + BOARD_SIZE * i);
			}
			auto cpu_te = high_resolution_clock::now();
			cpu_time += 0.001 * duration_cast<microseconds> (cpu_te - cpu_ts).count();
	
			for (int i = 0; i < boards_count; i++)
			{
				fwrite(cpu_solutions + BOARD_SIZE * i, sizeof(char), BOARD_SIZE, cpu_outputfile);
				fputc('\n', cpu_outputfile);
			}
	
			auto gpu_ts = high_resolution_clock::now();
			cudaStatus = solvewithGPU(boards, masks, boards_count, &gpu_copy_to_d_time, &gpu_alg_time, &gpu_copy_to_h_time);
			auto gpu_te = high_resolution_clock::now();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addWithCuda failed! %s\n", cudaGetErrorString(cudaStatus));
				return 1;
			}
			gpu_time += 0.001 * duration_cast<microseconds> (gpu_te - gpu_ts).count();
	
			for (int i = 0; i < boards_count; i++)
			{
				fwrite(boards + BOARD_SIZE * i, sizeof(char), BOARD_SIZE, gpu_outputfile);
				fputc('\n', gpu_outputfile);
			}
			done += boards_count;
			printf("Zrobilismy: %d\n", done);
			boards_count = 0;
		}
	}

	fputc('\0', cpu_outputfile);
	fputc('\0', gpu_outputfile);

	std::cout << "CPU Time:    " << setw(7) << cpu_time << " nsec" << endl;
	std::cout << "Whole GPU Time:    " << setw(7) << gpu_time << " nsec" << endl;
	std::cout << "Copy to device GPU time:    " << setw(7) << gpu_copy_to_d_time << " nsec" << endl;
	std::cout << "Algorithm GPU Time:    " << setw(7) << gpu_alg_time << " nsec" << endl;
	std::cout << "Copy to host GPU Time:    " << setw(7) << gpu_copy_to_h_time << " nsec" << endl;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	fclose(inputfile);
	fclose(cpu_outputfile);
	fclose(gpu_outputfile);

	return 0;
}

cudaError_t solvewithGPU(char* h_boards, uint16_t* h_masks, unsigned int boards, long long* copy_to_d_time, long long* alg_time, long long* copy_to_h_time)
{
	cudaError_t cudaStatus;

	char* d_boards;

	uint16_t* d_masks;

	int blocks = 0;
	if (boards % THREADS_IN_BLOCK != 0)
		blocks = ((boards - (boards % THREADS_IN_BLOCK)) / THREADS_IN_BLOCK) + 1;
	else
		blocks = boards / THREADS_IN_BLOCK;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	auto gpu_memory_alloc_ts = high_resolution_clock::now();
	cudaStatus = cudaMalloc(&d_boards, sizeof(char) * BOARD_SIZE * blocks * THREADS_IN_BLOCK * RECURSION_TREE_SIZE);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc d_boards failes!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_masks, sizeof(uint16_t) * MASK_ARRAY_SIZE * blocks * THREADS_IN_BLOCK * RECURSION_TREE_SIZE);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc d_masks failes!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_boards, h_boards, sizeof(char) * BOARD_SIZE * boards, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy d_boards failes!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_masks, h_masks, sizeof(uint16_t) * MASK_ARRAY_SIZE * blocks * THREADS_IN_BLOCK, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy d_masks failes!\n");
		goto Error;
	}

	auto gpu_memory_alloc_te = high_resolution_clock::now();
	*copy_to_d_time += 0.001 * duration_cast<microseconds>(gpu_memory_alloc_te - gpu_memory_alloc_ts).count();
	
	int numBlocks;
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the 
	//// maximum occupancy for a full device launch 
	//int gridSize;    // The actual grid size needed, based on input size 
	//
	//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, solve, 96, 0);
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, solve, 0, 0);
	// Round up according to array size 
	//gridSize = (arrayCount + blockSize - 1) / blockSize;


	auto gpu_sol_ts = high_resolution_clock::now();
	solve <<<blocks, THREADS_IN_BLOCK >>> (d_boards, d_masks, boards, RECURSION_TREE_SIZE);
	cudaStatus = cudaDeviceSynchronize();
	auto gpu_sol_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failes %s!\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	*alg_time += 0.001 * duration_cast<microseconds>(gpu_sol_te - gpu_sol_ts).count();

	auto gpu_memory_back_ts = high_resolution_clock::now();
	cudaStatus = cudaMemcpy(h_boards, d_boards, sizeof(char) * BOARD_SIZE * boards, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failes! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	auto gpu_memory_back_te = high_resolution_clock::now();
	*copy_to_h_time += 0.001 * duration_cast<microseconds>(gpu_memory_back_te - gpu_memory_back_ts).count();

Error:
	cudaFree(d_boards);
	cudaFree(d_masks);

	return cudaStatus;
}

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

__host__ bool setMasks(uint16_t* masks, char* board, int i)
{
	uint16_t localMasks[27] = { 0 };
	int not_zeroes_count = 0;
	for (int i = 0; i < BOARD_SIZE; i++)
	{
		int num = board[i] - '0';
		if (num != 0)
		{
			not_zeroes_count++;
			int res1 = (localMasks[GetRow(i)] >> (num - 1)) & 1;
			int res2 = (localMasks[9 + GetCol(i)] >> (num - 1)) & 1;
			int res3 = (localMasks[18 + GetBox(i)] >> (num - 1)) & 1;
			if ((localMasks[GetRow(i)] >> (num - 1)) & 1 || (localMasks[9 + GetCol(i)] >> (num - 1)) & 1 || (localMasks[18 + GetBox(i)] >> (num - 1)) & 1)
				return false;
			localMasks[GetRow(i)] |= 1 << (num - 1);
			localMasks[9 + GetCol(i)] |= 1 << (num - 1);
			localMasks[18 + GetBox(i)] |= 1 << (num - 1);
		}
	}

	if (not_zeroes_count >= 17)
	{
		memcpy(masks + i * MASK_ARRAY_SIZE, localMasks, sizeof(uint16_t) * 27);
		return true;
	}
	else
		return false;
}

__host__ __device__ uint8_t GetRow(uint8_t ind)
{
	return (ind - GetCol(ind)) / BOARD_DIM;
}

__host__ __device__ uint8_t GetCol(uint8_t ind)
{
	return ind % BOARD_DIM;
}

__host__ __device__ uint8_t GetBox(uint8_t ind)
{
	uint8_t row = GetRow(ind);
	uint8_t col = GetCol(ind);
	uint8_t boxrow = row - (row % BOX_DIM);
	uint8_t boxcol = (col - (col % BOX_DIM)) / BOX_DIM;
	return boxrow + boxcol;
}

__device__ void initializeStack(Stack* stack)
{
	stack->top = -1;
}

__device__ bool push(Stack* stack, uint8_t value)
{
	if (isFullStack(stack))
	{
		printf("Błąd: Stos jest pełny!\n");
		return false;
	}
	stack->data[++stack->top] = value;
	return true;
}

__device__ bool pop(Stack* stack, uint8_t* value)
{
	if (isEmptyStack(stack))
	{
		printf("Błąd: Stos jest pusty!\n");
		return false;
	}
	*value = stack->data[stack->top--];
	return true;
}

__device__ bool top(Stack* stack, uint8_t* value)
{
	if (isEmptyStack(stack))
	{
		printf("Błąd: Stos jest pusty!\n");
		return false;
	}
	*value = stack->data[stack->top];
	return true;
}

__device__ bool isEmptyStack(Stack* stack)
{
	return stack->top == -1;
}

__device__ bool isFullStack(Stack* stack)
{
	return stack->top == STACK_SIZE - 1;
}