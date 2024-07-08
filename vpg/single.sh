cd ~/postgraduate/vpg/process
nohup python trainer_hetero_single_gpu.py -g 0 -m "GAT" >> gat.log  2>&1 &
nohup python trainer_hetero_single_gpu.py -g 1 -m "GCN_GAT" >> gcn_gat.log  2>&1 &
nohup python trainer_hetero_single_gpu.py -g 2 -m "SAGE" >> sage.log  2>&1 &

wait

nohup python trainer_hetero_single_gpu.py -g 3 -m "GCN" >> gcn.log  2>&1 &
nohup python trainer_hetero_single_gpu.py -g 0 -m "SAGE_GAT" >> sage_gat.log  2>&1 &




