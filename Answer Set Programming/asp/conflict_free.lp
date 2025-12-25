% Program 1: Conflict-free sets
% Usage: clingo asp/conflict_free.lp <framework>.lp

% Guessing part: freely decide whether each argument is included.
{ in(A) } :- arg(A).

% Checking part: forbid attacking pairs inside the candidate set.
:- in(A), in(B), att(A,B).

#show in/1.
