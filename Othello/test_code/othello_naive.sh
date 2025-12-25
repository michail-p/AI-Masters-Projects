#!/bin/bash
# Script for the naive Othello program that you should beat. Don't forget to make it executable (chmod +x othello_navive)
# Note that this program does not use the time limit parameter, but always use depth 7 (to be easier to beat)

# usage: othello position 
# changes directory to the location of this script
cd "$(dirname "$0")"

# run the naive Othello 
java Othello $1
