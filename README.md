# ADS Recommendation algorithms 

## Table of Contents

1. Baseline and math infrastructure
2. Algorithms apply time depndent
3. Install Requiments
4. Run and output diagram

###1.  Baseline includes 3 algorithms

- SVD:          
```
j(0) = sima( r_{ui} - (q_i.T*p_u) ) + reg * ( ||qi||^2 + ||pu||^2)
```

- SVDPP:
```
 j(0) = sima( r_{ui} - ( M  +  bu + bi + q_i.T*p_u) ) + reg * ( ||qi||^2 + ||pu||^2 + ||bu||^2 + ||bi||^2 )
```
- SVDIDT:       
```
 j(0) = sima(r_{ui} - ( M  +  pu.T * qi_a  +  qi.T * pu_a   +   qi.T * pu ) ) + reg * ( ||qi||^2 + ||pu||^2 + ||pu_a||^2 + ||qi||^2 ) + pen * ( M - pu.T * qi )^2 )
```
###2. Conclusion & Future Work: Algorithm time dependent
`time dependent = sign(t - tu) * alpha * log(1 + |1-tu|^beta)`
```
j(0) = sima( r_{ui} - ( M  +  (pu + time_dependent).T * qi_a  +  qi.T * pu_a   +   qi.T * (pu + time_dependent) ) + reg * ( ||qi||^2 + ||pu||^2 + ||pu_a||^2 + ||qi||^2 ) + pen * ( M - (pu + time_dependent).T * qi )^2)
```
###3. Install Requiments

```
pip install -r requirements.txt
```
###4. Run code

```python
python baseline/run_experiment.py
```

![alt](rmse_training_200_epochs.png)