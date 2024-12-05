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

#define BOX_DIM 3 // Wymiar kwadracików w środku sudoku

#define BOARD_DIM 9 // BOX_DIM * BOX_DIM wymiar planszy

#define BOARD_SIZE 81 // BOARD_DIM * BOARD_DIM cała wielkość planszy

#define MAX_BOARDS 460800 // ilość plansz sudoku jakie przetworzy program (najlepiej wielokrotność THREADS_IN_BLOCK * MAX_BLOCKS)

#define MASK_ARRAY_SIZE 27 // wielkość tablicy masek potrzebnych na opisanie jednej planszy

#define RECURSION_TREE_SIZE 256 // ilość plansz jakie każdy wątek ma dla siebie do dyspozycji (najlepiej wielokrotność 8)

#define MAX_BLOCKS 240 // ilość bloków, które będą zakolejkowane przez karte

#define THREADS_IN_BLOCK 256 // ilość wątków w bloku (wielokrotność 32)

#define STACK_SIZE 40 // wielkość stacku dla wątku, który nie ma już wolnych tablic do dyspozycji

__host__ __device__ uint8_t cellmask_to_possibilities(uint16_t cellmask);
__host__ void print_board(char* board);
__host__ bool setMasks(uint16_t* masks, char* board, int i);
__host__ __device__ uint8_t GetRow(uint8_t ind);
__host__ __device__ uint8_t GetCol(uint8_t ind);
__host__ __device__ uint8_t GetBox(uint8_t ind);

// Struktura dla stosu
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

__device__ void setBit(uint8_t* controlArray, uint8_t index);
__device__ void clearBit(uint8_t* controlArray, uint8_t index);
__device__ uint8_t getBit(uint8_t* controlArray, uint8_t index);

__global__ void solve(char* boards, uint16_t* masks, const unsigned int size);

cudaError_t solvewithGPU(char* h_boards, uint16_t* h_masks, unsigned int boards, long long* copy_to_d_time, long long* alg_time, long long* copy_to_h_time);

/**
 * Funckja wykonująca algorytm rozwiązywania sudoku.
 *
 * @param boards - wskaźnik do tablicy, w której zapisane są plansze sudoku plus miejsca wolne na nowe plansze
 * @param masks - wskaźnik do tablicy, w której zapisane są maski dla plansz sudoku plus miejsca wolne na nowe maski dla nowych plansz
 * @param size - ilość tablic do rozwiązania
 */
__global__ void solve(char* boards, uint16_t* masks, const unsigned int size)
{
	char board[BOARD_SIZE]; // Lokalna tablica sudoku
	uint16_t locmasks[MASK_ARRAY_SIZE]; // Lokalna tablica masek (maski[9 masek dla rzędów, 9 masek dla kolumn, 9 masek dla kwadratów])
	Stack numStack, indStack; // Stosy dla liczb, które trzeba wpisać lub są wpisane w pola oraz stos indeksów, gdzie dana liczba jest lub powinna być wpisana 
	uint8_t controlArray[RECURSION_TREE_SIZE / 8]; // Tablica odpowiadająca za trzymanie informacji, która plansza sudoku w pamięci jest zapisana (każdy bit zapisuje stan jednej tablicy)
	setBit(controlArray, 0); // Ustawiamy, bit dla tablicy wejściowej
	initializeStack(&numStack);
	initializeStack(&indStack);
	uint8_t i = 0; // Indeks tablicy, którą aktualnie rozwiązujemy
	uint8_t j = (i + 1) % RECURSION_TREE_SIZE; // Indeks, od którego będziemy szukać pustej tablicy
	uint8_t status = 0; // To jest zamiast booli 1 - done rozpatrywanej tablicy, 2 - error rozpatrywanej tablicy

	memcpy(board, boards + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, sizeof(char) * BOARD_SIZE);
	memcpy(locmasks, masks + (blockDim.x * blockIdx.x + threadIdx.x) * MASK_ARRAY_SIZE, sizeof(uint16_t) * MASK_ARRAY_SIZE);

	while (status != 1)
	{
		// Jeżeli liczba kontrolna dla tej tablicy ma 0 to oznacza, że nie ma co rozwiązywać i musimy szukać następnej tablicy
		if (getBit(controlArray, i) != 1)
		{
			while (getBit(controlArray, i) != 1)
				i = (i + 1) % RECURSION_TREE_SIZE;
			memcpy(board, boards + blockDim.x * gridDim.x * BOARD_SIZE * i + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, sizeof(char) * BOARD_SIZE);
			memcpy(locmasks, masks + blockDim.x * gridDim.x * MASK_ARRAY_SIZE * i + (blockDim.x * blockIdx.x + threadIdx.x) * MASK_ARRAY_SIZE, sizeof(uint16_t) * MASK_ARRAY_SIZE);
			j = (i + 1) % RECURSION_TREE_SIZE;
		}

		uint8_t minimalpossibilities = 9;
		uint8_t minimalpossibilities_ind = 82;
		status = 1; // Status ten oznacza, że plansza skończona. W pętli, gdy napotkamy 0 zmieniamy ten status na 0, jako nie skończona.

		// Iteracja po wszystkich komórkach sudoku
		for (uint8_t k = 0; k < BOARD_SIZE; k++)
		{
			if (board[k] == '0')
			{
				status = 0; // Ustawiamy status, że jeszcze nie jest skończona ta plansza
				uint16_t mask = (locmasks[GetRow(k)] | locmasks[GetCol(k) + 9]) | locmasks[GetBox(k) + 18]; // Suma masek dla tej komórki

				// Jeżeli pierwsze 9 bitów zapełnionych, a dalej jest 0 w tej komórce, więc mamy błąd na planszy
				if (mask == 0x01FF)
				{
					status = 2;
					break;
				}
				// Jeżeli znaleźliśmy komórkę z mniejszą ilością wolnych liczb do nie do wpisania to ją zapamiętujemy
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

		// Udało się rozwiązać sudoku
		if (status == 1)
		{
			memcpy(boards + (blockDim.x * blockIdx.x + threadIdx.x) * BOARD_SIZE, board, sizeof(char) * BOARD_SIZE);
		}

		// Nie rozwiązaliśmy do końca sudoku, ale nie napotkaliśmy też jeszcze błędu
		if (status == 0)
		{
			uint8_t k = 0;
			// Maska dla danej komórki (suma masek z rzędu, kolumny i kwadracika)
			uint16_t combineMask = (locmasks[GetRow(minimalpossibilities_ind)] | locmasks[GetCol(minimalpossibilities_ind) + 9]) | locmasks[GetBox(minimalpossibilities_ind) + 18];

			// Iterujemy się po wszystkich 9 możliwych cyfrach w sudoku
			for (uint8_t l = 0; l < 9; l++)
			{
				if ((combineMask >> l) & 1) // Jeżeli jest 1 na pozycji l to znaczy, że już jest zajęta ta liczba i można iść dalej
					continue;

				if (k == minimalpossibilities - 1) // Jest to ostatnia możliwość, dlatego tą sobie do aktualnie rozważanej tablicy wpisujemy
				{
					// Wpisanie do tablicy cyfry
					board[minimalpossibilities_ind] = l + '0' + 1;

					// Aktualizacja masek
					locmasks[GetRow(minimalpossibilities_ind)] |= (1 << l);
					locmasks[GetCol(minimalpossibilities_ind) + 9] |= (1 << l);
					locmasks[GetBox(minimalpossibilities_ind) + 18] |= (1 << l);

					if (!isEmptyStack(&numStack)) // Jeżeli jest taka potrzeba to wrzucamy na stos, aby móc się potem wrócić
					{
						push(&numStack, l + 1);
						push(&indStack, minimalpossibilities_ind);
					}
					break;
				}

				while (j != i && status != 5)
				{
					if (getBit(controlArray,j) == 0) // Znaleziono pustą tablicę
						break;
					j = (j + 1) % RECURSION_TREE_SIZE;
				}

				if (j != i && status != 5) // Znaleziono pustą tablice w pamięci
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

				if (j == i && status != 5) // Nieznaleziono pustych tablica w pamięci
					status = 5;

				if (status == 5) // Nie ma pustych tablic więc musimy wrzucić na stos
				{
					push(&numStack, l + 1);
					push(&indStack, minimalpossibilities_ind);
				}

				k++;
			}
		}
	}
}

/**
 * Funkcja rozgrzewająca GPU (nie robi nic ciekawego).
 *
 * @param i - parametr ten nie ma znaczenia.
 */
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

	// Część rozgrzewająca GPU
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
	std::cout << "Copy to host GPU Time:    " << setw(7) << 0.001 * gpu_copy_to_h_time << " nsec" << endl;

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

/**
 * Funckja pomocnicza, agregująca wszystkie wywołania CUDA (alokacja pamięci na GPU, zapis danych do GPU, algorytm, odczyt wyników z GPU, zwolnienie pamięci)
 *
 * @param[in] h_boards - tablica, w której zapisane są plansze sudoku po stronie hosta
 * @param[in] h_masks - tablica, w której zapisane są maski dla plansz sudoku po stronie hosta
 * @param[in] boards - ilość tablic do rozwiązania
 * @param[out] copy_to_d_time - wskaźnik na zmienną z czasem, w jakim kopiujemy dane z hosta do GPU
 * @param[out] alg_time - wskaźnik na zmienną z czasem, w jakim wykonujemy funckję rozwiązywania sudoku (algorytm)
 * @param[out] copy_to_h_time - wskaźnik na zmienną z czasem, w jakim kopiujemy dane z GPU do hosta
 * @return możliwy error, który zaszedł podczas "CUDA-owych" operacji
 */
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

	auto gpu_sol_ts = high_resolution_clock::now();
	solve <<<blocks, THREADS_IN_BLOCK >>> (d_boards, d_masks, boards);
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
	*copy_to_h_time += duration_cast<microseconds>(gpu_memory_back_te - gpu_memory_back_ts).count();
	//printf("%d, %d\n", 0.001 * duration_cast<microseconds>(gpu_memory_back_te - gpu_memory_back_ts).count(), gpu_memory_back_ts);
Error:
	cudaFree(d_boards);
	cudaFree(d_masks);

	return cudaStatus;
}

/**
 * Zlicza cyfry jakie można wpisać w tą komórkę.
 *
 * @param cellmask Suma masek na danej komórce.
 * @return Liczba cyfr jakie mogą być wpisane w komórkę z tą maską.
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
 * Wypisuje planszę w konsoli.
 *
 * @param board Plansza do wypisania.
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

/**
 * Liczy, w którym rzędzie znajduje się komórka o danym indeksie.
 *
 * @param ind - indeks komórki
 * @return numer rzędu, w którym jest komórka (liczony od 0)
 */
__host__ __device__ uint8_t GetRow(uint8_t ind)
{
	return (ind - GetCol(ind)) / BOARD_DIM;
}

/**
 * Liczy, w której kolumnie znajduje się komórka o danym indeksie.
 *
 * @param ind - indeks komórki
 * @return numer kolumny, w której jest komórka (liczona od 0)
 */
__host__ __device__ uint8_t GetCol(uint8_t ind)
{
	return ind % BOARD_DIM;
}

/**
 * Liczy, w którym box-ie znajduje się komórka o danym indeksie.
 *
 * @param ind - indeks komórki
 * @return numer box-u, w którym jest komórka (liczony od 0 do 8, od lewego górnego rogu w prawo)
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
 * Inicjalizuje stos.
 *
 * @param stack - wskaźnik na stos do zainicjalizowania
 */
__device__ void initializeStack(Stack* stack)
{
	stack->top = -1;
}

/**
 * Wkłada na górę stosu przekazany element.
 *
 * @param stack - wskaźnik na stos
 * @param value - wartość, którą wrzucamy na stos
 * @return informacja o powodzeniu akcji, zwraca nie, gdy stos pełny
 */
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

/**
 * Zdejmuje z góry stosu element.
 *
 * @param[in] stack - wskaźnik na stos
 * @param[out] value -  wskaźnik na wartość, którą zdejmujemy ze stosu
 * @return informacja o powodzeniu akcji, zwraca nie, gdy stos pusty
 */
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

/**
 * Podgląda z góry stosu element.
 *
 * @param[in] stack - wskaźnik na stos
 * @param[out] value -  wskaźnik na wartość, którą podglądamy na stosie
 * @return informacja o powodzeniu akcji, zwraca nie, gdy stos pusty
 */
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

/**
 * Sprawdza, czy stos jest pusty.
 *
 * @param stack - wskaźnik na stos
 * @return informacja o "pustości"
 */
__device__ bool isEmptyStack(Stack* stack)
{
	return stack->top == -1;
}

/**
 * Sprawdza, czy stos jest pełny.
 *
 * @param stack - wskaźnik na stos
 * @return informacja o zapełnieniu stosu
 */
__device__ bool isFullStack(Stack* stack)
{
	return stack->top == STACK_SIZE - 1;
}

/**
 * Funkcja ustawiająca bit o indeksie podanym na 1.
 *
 * @param controlArray - wskaźnik do tablicy, w której zmieniamy wartość bitów
 * @param index - index zmienianego bita w tablicy
 */
__device__ void setBit(uint8_t* controlArray, uint8_t index)
{
	if (index < 0 || index >= RECURSION_TREE_SIZE)
		printf("Zły index!\n");
	else
		controlArray[index / 8] |= (1 << (index % 8));
}

/**
 * Funkcja ustawiająca bit o indeksie podanym na 0.
 *
 * @param controlArray - wskaźnik do tablicy, w której zmieniamy wartość bitów
 * @param index - index zmienianego bita w tablicy
 */
__device__ void clearBit(uint8_t* controlArray, uint8_t index)
{
	if (index < 0 || index >= RECURSION_TREE_SIZE)
		printf("Zły index!\n");
	else
		controlArray[index / 8] &= ~(1 << (index % 8));
}

/**
 * Funkcja pobierająca wartość bita na pozycji podanej przez index.
 *
 * @param controlArray - wskaźnik do tablicy, z której czytamy wartość bitów
 * @param index - index czytanego bita w tablicy
 * 
 * @returns Wartości bita, czyli 1 lub 0, albo 3, gdy został podany zły indeks i wystąpił błąd
 */
__device__ uint8_t getBit(uint8_t* controlArray, uint8_t index) {
	if (index < 0 || index >= RECURSION_TREE_SIZE)
	{
		printf("Zły index!\n");
		return 3;
	}
	return (controlArray[index / 8] & (1 << (index % 8))) != 0;
}
