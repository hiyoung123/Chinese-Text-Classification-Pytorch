python run.py --model TextCNN --save_by_step True --patience 1000
python run.py --model TextRNN --save_by_step True --patience 1000
python run.py --model TextRCNN --save_by_step True --patience 1000
python run.py --model DPCNN --save_by_step True --patience 1000
python run.py --model FastText --save_by_step True --patience 1000

python run.py --model BertFC --save_by_step True --patience 1000
python run.py --model BertCNN --save_by_step True --patience 1000
python run.py --model BertRNN --save_by_step True --patience 1000
python run.py --model BertRCNN --save_by_step True --patience 1000

python run.py --model TextCNN --save_by_step True --patience 1000 --init_weight True --task_name init_weight
python run.py --model TextRNN --save_by_step True --patience 1000 --init_weight True --task_name init_weight
python run.py --model TextRCNN --save_by_step True --patience 1000 --init_weight True --task_name init_weight
python run.py --model DPCNN --save_by_step True --patience 1000 --init_weight True --task_name init_weight
python run.py --model FastText --save_by_step True --patience 1000 --init_weight True --task_name init_weight

python run.py --model TextCNN --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss
python run.py --model TextRNN --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss
python run.py --model TextRCNN --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss
python run.py --model DPCNN --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss
python run.py --model FastText --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss

python run.py --model BertFC --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss
python run.py --model BertCNN --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss
python run.py --model BertRNN --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss
python run.py --model BertRCNN --save_by_step True --patience 1000 --loss_type focal_loss --task_name focal_loss

python run.py --model TextCNN --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth
python run.py --model TextRNN --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth
python run.py --model TextRCNN --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth
python run.py --model DPCNN --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth
python run.py --model FastText --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth

python run.py --model BertFC --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth
python run.py --model BertCNN --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth
python run.py --model BertRNN --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth
python run.py --model BertRCNN --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth

python run.py --model TextCNN --save_by_step True --patience 1000 --lookahead True --task_name lookahead
python run.py --model TextRNN --save_by_step True --patience 1000 --lookahead True --task_name lookahead
python run.py --model TextRCNN --save_by_step True --patience 1000 --lookahead True --task_name lookahead
python run.py --model DPCNN --save_by_step True --patience 1000 --lookahead True --task_name lookahead
python run.py --model FastText --save_by_step True --patience 1000 --lookahead True --task_name lookahead

python run.py --model BertFC --save_by_step True --patience 1000 --lookahead True --task_name lookahead
python run.py --model BertCNN --save_by_step True --patience 1000 --lookahead True --task_name lookahead
python run.py --model BertRNN --save_by_step True --patience 1000 --lookahead True --task_name lookahead
python run.py --model BertRCNN --save_by_step True --patience 1000 --lookahead True --task_name lookahead

python run.py --model BertFC --save_by_step True --patience 1000 --pre_trained_model pretrain/chinese_roberta_wwm_ext_pytorch --task_name roberta_base
python run.py --model BertCNN --save_by_step True --patience 1000 --pre_trained_model pretrain/chinese_roberta_wwm_ext_pytorch --task_name roberta_base
python run.py --model BertRNN --save_by_step True --patience 1000 --pre_trained_model pretrain/chinese_roberta_wwm_ext_pytorch --task_name roberta_base
python run.py --model BertRCNN --save_by_step True --patience 1000 --pre_trained_model pretrain/chinese_roberta_wwm_ext_pytorch --task_name roberta_base
