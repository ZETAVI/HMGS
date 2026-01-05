# 将训练的结果传输回我们的服务器中进行渲染
cd ~/autodl-tmp/HMGS && tar czf - output/ORIGINAL/bicycle | ssh -p 9997 lzb@43.142.58.25 "cd ~/projects/GSDF && tar xzf -"
tar czf - vgg16-397923af.pth | ssh -p 36529 root@connect.nma1.seetacloud.com "cd ~/.cache/torch/hub/checkpoints && tar xzf -"
# render
python ./render.py -m ./output/ORIGINAL/bicycle --config configs/mipnerf360/bicycle.yaml --skip_train --iteration -1 | tee output/ORIGINAL/bicycle/logger/render.log

# metrics
python metrics.py -m ./output/ORIGINAL/bicycle
