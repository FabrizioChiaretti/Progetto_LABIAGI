
# SUDOKU_SOLVER

**SUDOKU_SOLVER** is a tool whose purpose is to solve any Sudoku game using *genetic algorithms*.

## QUICK INSTALL GUIDE

1. Download the Github project on your machine;
2. Download python3 and opencv library on your machine;
3. Use the command *python3 Sudoku.py* to launch the tool.

## USER MANUAL

The "Sudoku" folder contains two images: "input.jpg" is the input of the tool while "solution.jpg" contains a solution to the game.
When launched, at first the program searches for the image "input.jpg". The tool processes that image to get the grid of the game and when finished it shows the result to the user, waiting until any key is pressed. 
When the user inputs a key, genetic algorithm starts solving the enigma finding a possible solution. When it finishes, it is necessary to asses that the output of the algorithm is effectively a solution to the game. Hence, using an image processing technique, the tool exctracts the solution from the image "solution.jpg" comparing it with the one found by the algorithm.

	 
