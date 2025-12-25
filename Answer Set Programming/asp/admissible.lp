% Program 2: Admissible sets
% Usage: clingo asp/admissible.lp <framework>.lp

% Guess: choose a conflict-free set as in Program 1.
{ in(A) } :- arg(A).

% Checking: no two in-arguments may attack each other.
:- in(A), in(B), att(A,B).

% Helper: arguments defeated/attacked by the current set.
defeated(B) :- in(A), att(A,B).

% Admissibility check: every argument inside must be defended.
:- in(A), att(B,A), not defeated(B).

#show in/1.
