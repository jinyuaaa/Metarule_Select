:- ['/home/jiny/Reinforcement_Meta_Rule_Learner/prolog/task_chess_jump_bk.pl'].

a :- Pos=[f([[0,0],[1,1],[2,2]],[[2,2],0,4]),
          f([[0,0],[1,1],[2,2],[1,1]],[[1,1],1,4]),
          f([[0,1],[1,2],[2,1],[1,0],[0,1]],[[0,1],0,4]),
          f([[1,1],[2,2]],[[2,2],1,4]),
          f([[2,0],[1,1],[0,2]],[[0,2],1,4]),
          f([[0,0],[1,1]],[[1,1],1,4])],
     Neg=[f([[0,1],[1,2],[2,1],[1,0],[0,1]],[[0,1],1,2]),
          f([[0,0],[1,1],[2,2]],[[2,2],0,3]),
          f([[1,1],[0,0]],[[1,0],1,4]),
          f([[2,1],[2,2],[1,2]],[[1,2],0,4])],
     learn(Pos,Neg).
