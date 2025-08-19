# CUDA Sudoku Solver

This project implements a **Sudoku solver in CUDA**, leveraging parallel threads to solve multiple Sudoku boards simultaneously.

Each thread is responsible for solving one Sudoku board using a combination of **minimum-possibility heuristics** and **recursive backtracking**.

## 🎓 Academic Context

This project was created as part of the academic course **Graphic Processors in Computational Applications** during the **2024/2025 winter semester** at **Warsaw University of Technology**.

The **CPU version** of the Sudoku solver was implemented by the lecturer and is used as a reference to compare both the **solving time** and the **correctness** of the GPU implementation.


## 🚀 Features

* Parallel solving of multiple Sudoku boards using CUDA
* Mask-based validation for rows, columns, and 3×3 squares
* Algorithm logic:

  1. Select the cell with the fewest possible candidates
  2. Fill the value immediately if there is only one option
  3. Branch recursively when multiple options exist (up to `RECURSION_TREE_SIZE`)
  4. Use backtracking with a stack of guesses when there is no space left in the memory for new board

## 🧩 Algorithm Overview

1. **Initialization**

   * Each thread gets its own Sudoku board
   * Masks are created (9 × rows, 9 × columns, 9 × 3×3 squares)
   * Initial mask state is stored in the mask array

2. **Solving Loop**

   * Continue until the board is solved
   * Find the cell with the smallest number of candidate values

3. **Value Insertion**

   * If there is **only one candidate** → insert and update masks
   * If there are **multiple candidates** →

     * If space is available in `RECURSION_TREE_SIZE` → spawn new boards with each candidate and updated masks
     * If no space is left → fall back to **backtracking** using a stack of guesses (indices and values)

## 📊 Parameters

Unfortunately all control parameters are currently **hardcoded** in the source file:

```c
#define RECURSION_TREE_SIZE 256   // Number of boards used for branching
#define MAX_BLOCKS 240            // Number of GPU blocks
#define THREADS_IN_BLOCK 256      // Threads per block
#define STACK_SIZE 100            // Stack size for backtracking when no recursion slots left
#define INPUTFILE_PATH "sudoku-sm.csv"    // Input file path
#define CPU_OUTPUTFILE_PATH "cpu_output.txt" // CPU results output
#define GPU_OUTPUTFILE_PATH "gpu_output.txt" // GPU results output
```

⚠️ **Note:** These values are not configurable at runtime — they must be changed directly in the code.

## 📈 Performance

### ✅ Strengths

* Fully parallelized – each CUDA thread solves a different Sudoku board
* Mask optimization enables fast possibility checking
* Hybrid approach: **recursive branching + backtracking** ensures efficiency and memory safety
* Parameters make it easy to optimize for different GPUs

### ⚠️ Limitations

* High thread divergence (some warps do significantly more work than others)
* Non-coalesced global memory access (threads do not read/write contiguous addresses)

**Observation:**
The program performs best on **large batches of relatively easy Sudoku boards**, but struggles to significantly outperform the CPU version when solving **harder boards**.

## 📄 Example Input (`sudoku-sm.csv`)

```
002610000000040290703000004290000400007000600100000820000380900000000305800005000
090700860031005020806000000007050006000307000500010700000000109020600350054008070
006004012000000305000503000002005000500060140017900000600000030000021700070050000
```

## 📄 Example Output (`gpu_output.txt`)

```
942617538618543297753298164295836471387124659164759823571382946426971385839465712
295743861431865927876192543387459216612387495549216738763524189928671354154938672
356874912741296385298513467962145873583762149417938256629487531835621794174359628
```

## 🔮 Future Improvements

* **Configurable parameters** – move hardcoded `#define` values to runtime arguments or configuration files
* **Better error handling and displaying** – detect unsolvable boards or invalid input
* **Optimized memory usage** – replace fixed-size arrays with dynamic allocation
* **Smarter heuristics** – experiment with alternative cell selection strategies to improve solving speed
* **Memory management** – fixing the non-coalesced memory access

## ⚙️ Requirements

* NVIDIA GPU with CUDA support
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## 📜 License

This project is released under the MIT License.
