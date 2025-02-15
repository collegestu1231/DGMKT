# DGMKT

![DGMKT](assets/DGMKT_overall.png)
This repository is built for the paper DGMKT: Leveraging Student Profiles and the Mamba Framework to Enhance Knowledge Tracing.
> **Author: mingxing Shao**, **tiancheng Zhang**

## Overview
This paper proposes a Dual-Graph Mamba framework for Knowledge Tracing (DGMKT), which models student profiles based on students’ interaction sequence
through a Dual-Graph Student-Profile Aware Module (DGSPM). Meanwhile, we model student mastery states based on Mamba, avoiding the
long-sequence forgetting problem in RNN-based models and the need for
bias functions in attention-based models for KT tasks. 

## Dataset 
We use four datasets(statics,assistment2017,assistment2009,kddcup2010) to demonstrate the effectiveness of proposed DGMKT. 

## Reproduce
if you want to reproduce DGMKT, please run the follow command to compile the files in csrc/selective_scan into 'selective_scan_cuda.cpython-38-x86_64-linux-gnu.so' (may be different up to your environments)
```{bash}
pip install . 
```


