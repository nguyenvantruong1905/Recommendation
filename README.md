# Recommendation

### Baseline includes 3 algorithms
- andrew:          
 j(0) = sima( r_{ui} - (q_i.T*p_u) ) + reg * ( ||qi||^2 + ||pu||^2)
- koren: SVD      
 j(0) = sima( r_{ui} - ( M  +  bu + bi + q_i.T*p_u) ) + reg * ( ||qi||^2 + ||pu||^2 + ||bu||^2 + ||bi||^2 )
- edulive:       
 j(0) = sima( r_{ui} - ( M  +  pu.T * qi_a  +  qi.T * pu_a   +   qi.T * pu ) ) + reg * ( ||qi||^2 + ||pu||^2 + ||pu_a||^2 + ||qi||^2 ) + pen * ( M - pu.T * qi )^2)
### Algorithm implicit info and time dependent
### implicit :
- homework time (time_exam)
- number of operations on screen(action_exam)
## time dependent :
- homework day
##### time dependent = sign(t - tu) * alpha * log(1 + |1-tu|^beta) 
j(0) = sima( r_{ui} - ( M  +  (pu + time_dependent).T * qi_a  +  qi.T * pu_a   +   qi.T * (pu + time_dependent) - l1 * time_exam - l2 * action_exam) ) + reg * ( ||qi||^2 + ||pu||^2 + ||pu_a||^2 + ||qi||^2 ) + pen * ( M - (pu + time_dependent).T * qi )^2)
