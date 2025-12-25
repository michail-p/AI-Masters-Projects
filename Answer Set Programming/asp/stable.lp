% Program 3: Stable extensions
% Usage: clingo asp/stable.lp <framework>.lp

% Guess conflict-free candidate.
{ in(A) } :- arg(A).
:- in(A), in(B), att(A,B).

% Mark everything attacked by the candidate set.
defeated(B) :- in(A), att(A,B).

% Stable requirement: every argument outside must be attacked.
:- arg(A), not in(A), not defeated(A).

#show in/1.
