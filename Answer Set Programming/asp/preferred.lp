% Program 4: Preferred extensions (maximal admissible sets)
% Source: Adapted from the ASPARTIX "prefex_gringo.lp" encoding
% https://www.dbai.tuwien.ac.at/research/argumentation/aspartix/dung/prefex_gringo.lp
% Usage: clingo asp/preferred.lp <framework>.lp

% Guess a set S subseteq AR
in(X) :- not out(X), arg(X).
out(X) :- not in(X), arg(X).

% S must be conflict-free
:- in(X), in(Y), att(X,Y).

% defeated(X) iff S attacks X
defeated(X) :- in(Y), att(Y,X).

% not_defended(X) iff some attacker of X is not defeated by S
not_defended(X) :- att(Y,X), not defeated(Y).

% Admissibility: every member of S is defended
:- in(X), not_defended(X).

% ------------------------------------------------------------
% Ordering over the domain (successor relation with infimum/supremum)
% ------------------------------------------------------------

lt(X,Y) :- arg(X), arg(Y), X < Y.
nsucc(X,Z) :- lt(X,Y), lt(Y,Z).
succ(X,Y) :- lt(X,Y), not nsucc(X,Y).
ninf(X) :- lt(Y,X).
nsup(X) :- lt(X,Y).
inf(X) :- not ninf(X), arg(X).
sup(X) :- not nsup(X), arg(X).

% Guess S' superset S
inN(X) :- in(X).
inN(X) | outN(X) :- out(X).

% Determine whether S' equals S using successor walk
eq_upto(Y) :- inf(Y), in(Y), inN(Y).
eq_upto(Y) :- inf(Y), out(Y), outN(Y).
eq_upto(Y) :- succ(Z,Y), in(Y), inN(Y), eq_upto(Z).
eq_upto(Y) :- succ(Z,Y), out(Y), outN(Y), eq_upto(Z).
eq :- sup(Y), eq_upto(Y).

% Track arguments not defeated by S'
undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y).
undefeated_upto(X,Y) :- inf(Y), outN(X), not att(Y,X).
undefeated_upto(X,Y) :- succ(Z,Y), undefeated_upto(X,Z), outN(Y).
undefeated_upto(X,Y) :- succ(Z,Y), undefeated_upto(X,Z), not att(Y,X).
undefeated(X) :- sup(Y), undefeated_upto(X,Y).

% Spoil flag triggered whenever S' is unsuitable
not_empty :- arg(X).
spoil :- not not_empty.
spoil :- eq.
spoil :- inN(X), inN(Y), att(X,Y).
spoil :- inN(X), outN(Y), att(Y,X), undefeated(Y).

% Saturate witness guess once spoil holds
inN(X) :- spoil, arg(X).
outN(X) :- spoil, arg(X).

% Require spoil to hold (i.e., no admissible strict superset exists)
:- not spoil.

#show in/1.
